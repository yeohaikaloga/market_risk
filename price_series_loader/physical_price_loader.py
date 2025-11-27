from price_series_loader.price_series_loader import PriceLoader
import pandas as pd
from sqlalchemy import text


class PhysicalPriceLoader(PriceLoader):

    def load_prices(self, start_date, end_date, data_source, reindex_dates=None, instrument_id=None) -> pd.DataFrame:
        instrument_id = instrument_id or self.instrument_id
        grade = self.params.get('grade')
        origin = self.params.get('origin')
        type_ = self.params.get('type')
        external_name_filter = f"pc.external_name = '{instrument_id}'"
        if instrument_id == 'A Index':
            external_name_filter = f"pc.external_name LIKE '%A Index'"

        # TODO DISTINCT should not have to be there...
        query = f"""
            SELECT DISTINCT pmp.tdate, pmp.px_settle, pc.external_name, pc.shipment_month, 
                   pi.origin, pi.type, pi.grade, pi.crop_year, pi.crop_year_label
            FROM ref.physical_contract pc 
            JOIN market.physical_market_price pmp 
                ON pc.traded_contract_id = pmp.traded_contract_id
            JOIN ref.physical_instrument pi 
                ON pi.id = pc.instrument_id
            JOIN ref.product p 
                ON pi.product_id = p.id 
            WHERE pmp.data_source = '{data_source}'
            AND {external_name_filter}
            AND pmp.tdate BETWEEN '{start_date}' AND '{end_date}'
        """
        if grade:
            query += f" AND pi.grade = '{grade}'"
        if origin:
            query += f" AND pi.origin = '{origin}'"
        if type_:
            query += f" AND pi.type = '{type_}'"

        print(query)

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        df['tdate'] = pd.to_datetime(df['tdate'], errors='coerce')
        df.sort_values('tdate', inplace=True)

        duplicates_check = df.groupby(['tdate', 'crop_year']).size().reset_index(name='count')
        duplicates_check = duplicates_check[duplicates_check['count'] > 1]

        if not duplicates_check.empty:
            raise ValueError(f"Duplicate rows found for (tdate, crop_year):\n{duplicates_check}")

        print("Validation passed: No duplicate (tdate, crop_year) pairs.")

        if reindex_dates is not None:
            df = df.set_index('tdate').reindex(reindex_dates).reset_index()

        return df

    def load_ex_gins6_prices_from_staging(self, start_date, end_date, reindex_dates=None) -> pd.DataFrame:

        query = f"""
        SELECT * FROM staging.cotton_shankar6_price
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        """

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if reindex_dates is not None:
            df = df.set_index('tdate').reindex(reindex_dates).reset_index()

        return df
