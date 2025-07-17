from contract.futures_contract import FuturesContract
from generated_price_series.generated_price_series import PriceSeriesGenerator
import pandas as pd


class GenericCurveGenerator(PriceSeriesGenerator):

    def __init__(self, df: pd.DataFrame, futures_contract: FuturesContract):
        super().__init__(df)
        self.df = df
        self.futures_contract = futures_contract

    def validate_data(self):
        if self.price_df.empty:
            raise ValueError("Price DataFrame is empty.")

    def get_active_contracts(self, date):
        row = self.price_df.loc[date]
        return row[row.notna()]

    def calculate_roll_dates(self, roll_days: int) -> dict:
        roll_dates = {}
        for contract in self.contracts:
            valid_dates = self.price_df[contract].dropna().index
            if not valid_dates.empty:
                last_date = valid_dates[-1]
                roll_idx = max(0, len(valid_dates) - 1 - roll_days)
                roll_dates[contract] = valid_dates[roll_idx]
        return roll_dates

    def generate_generic_curve(self, position: int = 1, roll_days: int = 0, adjustment: str = 'none') -> pd.DataFrame:
        """
        Generate the N-th generic futures curve with optional early rolling and backward adjustment.

        Parameters:
            self (pd.DataFrame): DataFrame with datetime index and futures contract columns (e.g., CTH4, CTK4).
            position (int): Generic position to compute (1 = front, 2 = second, etc.).
            roll_days (int): Number of business days before a contract's last availability to roll.
            adjustment (str): 'none', 'ratio', or 'difference'.

        Returns:
        pd.DataFrame: DataFrame indexed by date with columns:
            - 'price': The unadjusted price from the selected contract on each date.
            - 'contract_to_use': The contract symbol selected on each date.
            - 'final_price': The adjusted price series if adjustment is applied; otherwise same as 'price'.
            - 'adjustment_values': The cumulative adjustment factor or difference applied on each date.
        """

        df = self.df.sort_index(ascending=True)
        contracts = df.columns.tolist()
        index = df.index
        generic_curve = pd.DataFrame(index=index, columns=['final_price', 'contract_to_use', 'price', 'adjustment_values'])

        # Load expiry and roll dates
        contract_expiry_dates = self.futures_contract.load_expiry_dates()
        contract_roll_dates = {k: v - pd.Timedelta(days=roll_days) for k, v in contract_expiry_dates.items()}
        print(contract_expiry_dates)
        print(contract_roll_dates)

        # STEP 1: Build unadjusted generic curve
        for date in index:
            eligible_contracts = []
            for contract in contracts:
                price = df.at[date, contract]
                if pd.notna(price):
                    roll_date = contract_roll_dates[contract]
                    expiry_date = contract_expiry_dates[contract]
                    if roll_date is None or expiry_date is None:
                        continue  # skip if info missing

                    if date <= roll_date:
                        eligible_contracts.append(contract)

            if len(eligible_contracts) >= position:
                contract_to_use = eligible_contracts[position - 1]
                price = df.at[date, contract_to_use]
                generic_curve.at[date, 'price'] = price
                generic_curve.at[date, 'contract_to_use'] = contract_to_use
            else:
                generic_curve.at[date, 'price'] = pd.NA
                generic_curve.at[date, 'contract_to_use'] = None

        # STEP 2: Apply backward adjustment if requested
        if adjustment.lower() in {'ratio', 'difference'}:
            prev_contract = None
            cumulative_ratio = 1.0
            cumulative_diff = 0.0
            adjustment_values = {}
            adjusted_prices = {}

            prev_contract = None
            prev_date = None

            for date in reversed(index):
                contract = generic_curve.at[date, 'contract_to_use']
                price = generic_curve.at[date, 'price']

                if pd.isna(price) or contract is None:
                    adjusted_prices[date] = pd.NA
                    adjustment_values[date] = pd.NA
                    prev_contract = None
                    prev_date = None
                    continue

                # When contract changes (i.e., rolling happened)
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

        return generic_curve