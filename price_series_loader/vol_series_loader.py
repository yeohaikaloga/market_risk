from price_series_loader.price_series_loader import PriceLoader
from utils.contract_utils import custom_monthly_contract_sort_key
import pandas as pd
from sqlalchemy import text



class VolLoader(PriceLoader):

    def __init__(self, instrument_name, source):
        super().__init__(instrument_name, source)
        self.price_history = None
        self.instrument_id = instrument_name
        self.contracts: list[str] | None = None
        self.source = source

    def load_prices(self, start_date, end_date):
        """Load historical price_series_loader data into price_history."""
        print('load price_series_loader')
        pass

    def load_vol_change_for_generic_curve(self, start_date, end_date, max_generic_curve=None, reindex_dates=None, instrument_name=None):
        if len(instrument_name) == 1:
            instrument_name = instrument_name + ' '
        generic_curve_list = [f"{instrument_name}{i}" for i in range(1, max_generic_curve + 1)]
        generic_curve_list_formatted = "(" + ", ".join(f"'{x}'" for x in generic_curve_list) + ")"
        print('formatted:', generic_curve_list_formatted)

        query = f"""
                SELECT * 
                FROM var_cont_vol_change_table_view
                WHERE cont_month IN {generic_curve_list_formatted}
                AND settlement_date BETWEEN '{start_date}' AND '{end_date}'
            """
        print(query)
        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        df['settlement_date'] = pd.to_datetime(df['settlement_date'])
        vol_change_df = df.groupby(['cont_month', 'settlement_date'])['vol_change'].last().unstack(level=0)

        return vol_change_df