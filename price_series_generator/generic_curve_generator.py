from contract_ref_loader.derivatives_contract_ref_loader import DerivativesContractRefLoader
from price_series_generator.price_series_generator import PriceSeriesGenerator
import pandas as pd


class GenericCurveGenerator(PriceSeriesGenerator):

    def __init__(self, df: pd.DataFrame, futures_contract: DerivativesContractRefLoader):
        super().__init__(df)
        self.df = df
        self.futures_contract = futures_contract
        self._contract_expiry_dates = None
        self._contract_roll_dates = None

    def _prepare_curve_inputs(self, roll_days: int):
        """
        Preload expiry and roll dates once per instrument.

        Logic:
        - Expiry dates are loaded directly from the instrument reference and remain
          unchanged (exchange-defined contractual expiry).
        - Roll dates are initially computed as:  roll_date = expiry_date - roll_days.
        - If the computed roll date falls on an Opera-defined market holiday
          (based on the instrument's holiday calendar), the roll date is shifted
          backward day-by-day until it lands on a valid trading day.
          Only roll dates are adjusted; expiry dates remain fixed.

        This ensures that the generic curve does not attempt to roll on a date with
        no market data available (holiday), while still respecting the official
        expiry schedule.
        """
        if self._contract_expiry_dates is None or self._contract_roll_dates is None:

            # Load expiries
            expiry = self.futures_contract.load_underlying_futures_expiry_dates(mode='futures')

            # Initial (raw) roll dates = expiry minus N days
            roll = {k: v - pd.Timedelta(days=roll_days) for k, v in expiry.items()}

            # --- Holiday Adjustment ---
            roll_dates_adj = {}
            calendar = self.futures_contract.holiday_calendar

            for contract, rd in roll.items():
                adj_rd = rd

                # If rd is a holiday, shift backward until we hit a business day
                while adj_rd in calendar.index and calendar.loc[adj_rd, "is_holiday"] == 1:
                    adj_rd -= pd.Timedelta(days=1)

                roll_dates_adj[contract] = adj_rd

            # Save attributes
            self._contract_expiry_dates = expiry
            self._contract_roll_dates = roll_dates_adj

    def generate_generic_curve(self, position: int = 1, roll_days: int = 0, adjustment: str = None,
                               usd_conversion_mode: str = None, forex_mode: str = None, fx_spot_df: pd.DataFrame = None,
                               cob_date: str = None) -> pd.DataFrame:
        """
        Generate the N-th generic futures curve with optional early rolling and backward adjustment.

        Parameters:
            self (pd.DataFrame): DataFrame with datetime index and futures contract_ref_loader columns (e.g., CTH4,
            CTK4).
            position (int): Generic position_loader to compute (1 = front, 2 = second, etc.).
            roll_days (int): Number of business days before a contract_ref_loader's last availability to roll.
            adjustment (str): None, 'ratio', or 'difference'.
            usd_conversion_mode (str): None, 'pre', or 'post'

        Returns:
        pd.DataFrame: DataFrame indexed by date with columns:
            - 'price_series_loader': The unadjusted price_series_loader from the selected contract_ref_loader on each
            date.
            - 'active_contract': The contract_ref_loader symbol selected on each date.
            - 'final_price': The adjusted price_series_loader series if adjustment is applied; otherwise same as
            'price_series_loader'.
            - 'adjustment_values': The cumulative adjustment factor or difference applied on each date.
        """
        valid_adjustments = {None, 'ratio', 'difference'}
        if adjustment not in valid_adjustments:
            raise ValueError(f"Invalid adjustment method: {adjustment}. Must be None or {valid_adjustments}")

        self._prepare_curve_inputs(roll_days)
        df = self.df.copy()

        if usd_conversion_mode == 'pre':
            if fx_spot_df is None:
                raise ValueError("fx_rates DataFrame must be provided for USD conversion")

            # Identify currency of the instrument from the futures contract
            instrument_currency = self.futures_contract.currency
            if instrument_currency.upper() == 'USD':
                pass  # already USD
            else:
                fx_rate = 'USD' + instrument_currency.upper()
                if fx_rate not in fx_spot_df.columns:
                    raise ValueError(f"FX rates for {fx_rate} not found in fx_rates DataFrame")

                if forex_mode == 'cob_date_fx':
                    fx_rate = fx_spot_df.at[cob_date, fx_rate]
                    df = df.div(fx_rate, axis=0)

                elif forex_mode == 'daily_fx':
                    df = df.div(fx_spot_df[fx_rate], axis=0)

                else:
                    raise ValueError(f"forex mode for {forex_mode} not found in.")

        df = df.interpolate(method='linear', limit_direction='both', axis=0)  # interpolate after forex

        contracts = df.columns.tolist()
        print('columns:', contracts)
        index = df.index
        generic_curve = pd.DataFrame(index=index,
                                     columns=['final_price',
                                              'active_contract',
                                              'price_series_loader',
                                              'adjustment_values'])

        # Load expiry and roll dates
        expiry_dates = self._contract_expiry_dates
        roll_dates = self._contract_roll_dates
        print('expiry:', expiry_dates)
        print('roll:', roll_dates)

        # STEP 1: Build unadjusted generic curve
        for date in index:
            eligible_contracts = []
            for contract in contracts:
                full_contract_name = contract + ' Comdty'
                if full_contract_name not in roll_dates.keys():
                    full_contract_name = contract + ' COMB Comdty'
                price = df.at[date, contract]
                if pd.notna(price):
                    roll_date = roll_dates[full_contract_name]
                    expiry_date = expiry_dates[full_contract_name]
                    if roll_date is None or expiry_date is None:
                        continue  # skip if info missing

                    if date <= roll_date:
                        eligible_contracts.append(full_contract_name)

            if len(eligible_contracts) >= position:
                active_contract = eligible_contracts[position - 1].replace(' COMB', '').replace(' Comdty', '')
                price = df.at[date, active_contract]
                generic_curve.at[date, 'price_series_loader'] = price
                generic_curve.at[date, 'active_contract'] = active_contract
            else:
                generic_curve.at[date, 'price_series_loader'] = pd.NA
                generic_curve.at[date, 'active_contract'] = None

        print(generic_curve.head())
        # STEP 2: Apply backward adjustment if requested
        if adjustment.lower() in {'ratio', 'difference'}:
            cumulative_ratio = 1.0
            cumulative_diff = 0.0
            adjustment_values = {}
            adjusted_prices = {}
            prev_contract = None

            for date in reversed(index):
                contract = generic_curve.at[date, 'active_contract']
                price = generic_curve.at[date, 'price_series_loader']

                if pd.isna(price) or contract is None:
                    adjusted_prices[date] = pd.NA
                    adjustment_values[date] = pd.NA
                    prev_contract = None
                    continue

                # When contract_ref_loader changes (i.e., rolling happened)
                if prev_contract and contract != prev_contract:
                    # Get prices on roll date (same date) for both contracts
                    from_price = df.at[date, prev_contract] if prev_contract in df.columns else None
                    to_price = df.at[date, contract] if contract in df.columns else None
                    if pd.notna(from_price) and pd.notna(to_price):
                        if adjustment == 'ratio' and to_price != 0:
                            ratio = from_price / to_price
                            cumulative_ratio *= ratio
                        elif adjustment == 'difference':
                            diff = from_price - to_price
                            cumulative_diff += diff

                # Apply adjustment
                if adjustment == 'ratio':
                    adjusted_price = price * cumulative_ratio
                    adjustment_value = cumulative_ratio
                elif adjustment == 'difference':
                    adjusted_price = price + cumulative_diff
                    adjustment_value = cumulative_diff
                else:
                    adjusted_price = price
                    adjustment_value = None

                adjusted_prices[date] = adjusted_price
                adjustment_values[date] = adjustment_value
                prev_contract = contract

            # Add adjusted prices to generic_curve DataFrame
            generic_curve['final_price'] = pd.Series(adjusted_prices)
            generic_curve['adjustment_values'] = pd.Series(adjustment_values)

        else:
            generic_curve['final_price'] = generic_curve['price_series_loader']
            generic_curve['adjustment_values'] = pd.NA

        if usd_conversion_mode == 'post':
            if fx_spot_df is None:
                raise ValueError("fx_spot_df DataFrame must be provided for USD conversion")

            if cob_date is None:
                raise ValueError("cob_date must be provided for post USD conversion")

            instrument_currency = self.futures_contract.currency

            if instrument_currency.upper() == 'USD':
                # Already in USD
                generic_curve['final_price_USD'] = generic_curve['final_price']
            else:
                fx_rate = 'USD' + instrument_currency.upper()
                if fx_rate not in fx_spot_df.columns:
                    raise ValueError(f"FX rates for {fx_rate} not found in fx_spot_df")

                # Use only the FX rate on cob_date
                if cob_date not in fx_spot_df.index:
                    raise ValueError(f"COB date {cob_date} not found in fx_spot_df index")

                if forex_mode == 'cob_date_fx':
                    fx_rate = fx_spot_df.at[cob_date, fx_rate]
                    generic_curve['fx_rate'] = fx_rate

                elif forex_mode == 'daily_fx':
                    generic_curve['fx_rate'] = fx_spot_df[fx_rate]

                else:
                    raise ValueError(f"forex mode for {forex_mode} not found in.")

                generic_curve['final_price_USD'] = generic_curve['final_price'].div(generic_curve['fx_rate'], axis=0)
        else:
            generic_curve['final_price_USD'] = generic_curve['final_price']

        # Fill any remaining NaNs forward
        generic_curve = generic_curve.ffill()
        return generic_curve

    def generate_generic_curves_df_up_to(self, max_position: int, roll_days: int = 14,
                                         adjustment: str = None, label_prefix: str = '',
                                         usd_conversion_mode: str = None, forex_mode: str = None,
                                         fx_spot_df = pd.DataFrame, cob_date: str = None) -> (
            tuple)[pd.DataFrame, pd.DataFrame]:
        """

        Generate and combine multiple generic curves (e.g., G1, G2, G3...) into one DataFrame.

        Parameters:
            max_position (int): Highest generic curve position_loader to generate (inclusive).
            roll_days (int): Days before expiry to roll.
            adjustment (str): 'none', 'ratio', or 'difference'.
            label_prefix (str): Optional prefix for column labels (e.g., instrument name).

        Returns:
        Tuple of:
            - Combined DataFrame of adjusted prices (final_price) for each generic curve
            - Combined DataFrame of active contract names tagged to each curve on each date
        """
        combined_df = pd.DataFrame(index=self.df.index)
        active_contracts_df = pd.DataFrame(index=self.df.index)

        for pos in range(1, max_position + 1):
            curve = self.generate_generic_curve(position=pos,
                                                roll_days=roll_days,
                                                adjustment=adjustment,
                                                usd_conversion_mode=usd_conversion_mode,
                                                forex_mode=forex_mode,
                                                fx_spot_df=fx_spot_df,
                                                cob_date=cob_date)
            if len(label_prefix) == 1:
                label_prefix = label_prefix + ' '
            col_name = f"{label_prefix}{pos}" if label_prefix else str(pos)
            combined_df[col_name] = curve['final_price_USD']
            active_contracts_df[col_name] = curve['active_contract']

        return combined_df, active_contracts_df

# TODO Validate the generic curves against BBG generated curves
