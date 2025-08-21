from contract_ref_loader.derivatives_contract_ref_loader import DerivativesContractRefLoader
from price_series_generator.price_series_generator import PriceSeriesGenerator
import pandas as pd


class GenericCurveGenerator(PriceSeriesGenerator):

    def __init__(self, df: pd.DataFrame, futures_contract: DerivativesContractRefLoader):
        super().__init__(df)
        self.df = df
        self.futures_contract = futures_contract

    def generate_generic_curve(self, position: int = 1, roll_days: int = 0, adjustment: str = 'none') -> pd.DataFrame:
        """
        Generate the N-th generic futures curve with optional early rolling and backward adjustment.

        Parameters:
            self (pd.DataFrame): DataFrame with datetime index and futures contract_ref_loader columns (e.g., CTH4, CTK4).
            position (int): Generic position_loader to compute (1 = front, 2 = second, etc.).
            roll_days (int): Number of business days before a contract_ref_loader's last availability to roll.
            adjustment (str): 'none', 'ratio', or 'difference'.

        Returns:
        pd.DataFrame: DataFrame indexed by date with columns:
            - 'price_series_loader': The unadjusted price_series_loader from the selected contract_ref_loader on each date.
            - 'active_contract': The contract_ref_loader symbol selected on each date.
            - 'final_price': The adjusted price_series_loader series if adjustment is applied; otherwise same as
            'price_series_loader'.
            - 'adjustment_values': The cumulative adjustment factor or difference applied on each date.
        """


        valid_adjustments = {'none', 'ratio', 'difference'}
        if adjustment not in valid_adjustments:
            raise ValueError(f"Invalid adjustment method: {adjustment}. Must be one of {valid_adjustments}")

        df = self.df.sort_index(ascending=True)
        contracts = df.columns.tolist()
        print('columns:', contracts)
        index = df.index
        generic_curve = pd.DataFrame(index=index,
                                     columns=['final_price', 'active_contract', 'price_series_loader',
                                              'adjustment_values'])

        # Load expiry and roll dates
        contract_expiry_dates = self.futures_contract.load_underlying_futures_expiry_dates(mode='futures')
        contract_roll_dates = {k: v - pd.Timedelta(days=roll_days) for k, v in contract_expiry_dates.items()}
        print('expiry:', contract_expiry_dates)
        print('roll:', contract_roll_dates)

        # STEP 1: Build unadjusted generic curve
        for date in index:
            eligible_contracts = []
            for contract in contracts:
                full_contract_name = contract + ' Comdty'
                price = df.at[date, contract]
                if pd.notna(price):
                    roll_date = contract_roll_dates[full_contract_name]
                    expiry_date = contract_expiry_dates[full_contract_name]
                    if roll_date is None or expiry_date is None:
                        continue  # skip if info missing

                    if date <= roll_date:
                        eligible_contracts.append(full_contract_name)

            if len(eligible_contracts) >= position:
                active_contract = eligible_contracts[position - 1].replace(" Comdty","")
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
            prev_date = None

            for date in reversed(index):
                contract = generic_curve.at[date, 'active_contract']
                price = generic_curve.at[date, 'price_series_loader']

                if pd.isna(price) or contract is None:
                    adjusted_prices[date] = pd.NA
                    adjustment_values[date] = pd.NA
                    prev_contract = None
                    prev_date = None
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
                prev_date = date

            # Add adjusted prices to generic_curve DataFrame
            generic_curve['final_price'] = pd.Series(adjusted_prices)
            generic_curve['adjustment_values'] = pd.Series(adjustment_values)

        else:
            # No adjustment: just copy
            generic_curve['final_price'] = generic_curve['price_series_loader']
            generic_curve['adjustment_values'] = pd.NA

        return generic_curve

    def generate_generic_curves_df_up_to(self, max_position: int, roll_days: int = 14,
                                         adjustment: str = 'none', label_prefix: str = '') \
            -> tuple[pd.DataFrame, pd.DataFrame]:
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
            curve = self.generate_generic_curve(position=pos, roll_days=roll_days, adjustment=adjustment)
            col_name = f"{label_prefix}{pos}" if label_prefix else str(pos)
            combined_df[col_name] = curve['final_price']
            active_contracts_df[col_name] = curve['active_contract']

        return combined_df, active_contracts_df
