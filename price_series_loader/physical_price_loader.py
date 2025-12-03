from price_series_loader.price_series_loader import PriceLoader
import pandas as pd
from sqlalchemy import text


class PhysicalPriceLoader(PriceLoader):

    def load_prices(self, start_date, end_date, data_source, reindex_dates=None, params=None) -> pd.DataFrame:

        # TODO DISTINCT should not have to be there...
        # TODO Is a product_name split necessary? Ideally it should be filtered by instrument_id; pending correction of Cotlook Index instrument_id
        if self.params.get('product_name') == 'Cotton':
            if self.params.get('instrument_name') == 'A Index':
                external_name_filter = f"pc.external_name LIKE '%A Index'"
            else:
                external_name_filter = f"pc.external_name = '{self.params.get('external_name')}'"
            cotton_query = f"""
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
            if self.params.get('grade'):
                cotton_query += f" AND pi.grade = '{self.params.get('grade')}'"
            if self.params.get('origin'):
                cotton_query += f" AND pi.origin = '{self.params.get('origin')}'"
            if self.params.get('type'):
                cotton_query += f" AND pi.type = '{self.params.get('type')}'"

            print(cotton_query)
            with self.source.connect() as conn:
                df = pd.read_sql_query(text(cotton_query), conn)

        elif self.params.get('product_name') == 'Wood':
            wood_query = f"""
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
                AND pc.instrument_id = {self.params.get('instrument_id')}
                AND pmp.tdate BETWEEN '{start_date}' AND '{end_date}'
                """

            with self.source.connect() as conn:
                df = pd.read_sql_query(text(wood_query), conn)

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

    #TODO Load from master prices table when ready
    def load_ex_gin_s6_price_from_staging(self, start_date, end_date, reindex_dates=None) -> pd.DataFrame:

        query = f"""
        SELECT * FROM staging.cotton_shankar6_price
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        """

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if reindex_dates is not None:
            df = df.set_index('tdate').reindex(reindex_dates).reset_index()

        return df

    def load_garmmz_sugar_price_from_staging(self, start_date, end_date, reindex_dates=None) -> pd.DataFrame:

        query = f"""
        SELECT * FROM staging.bio_cane_price
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        AND product = 'Sugar'
        """

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if reindex_dates is not None:
            df = df.set_index('tdate').reindex(reindex_dates).reset_index()

        return df

    def load_maize_up_price_from_staging(self, start_date, end_date, reindex_dates=None) -> pd.DataFrame:

        query = f"""
        SELECT * FROM staging.bio_cane_price
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        AND product = 'Maize'
        """

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if reindex_dates is not None:
            df = df.set_index('tdate').reindex(reindex_dates).reset_index()

        return df