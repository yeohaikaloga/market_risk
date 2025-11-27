from contract_ref_loader.contract_ref_loader import ContractRefLoader
import pandas as pd
from sqlalchemy import text

crop_dict = {'Brazilian': {'grade': 'M_1-1/8"_std', 'type': None, 'product_name': 'Cotton'},
             'Burkina Faso Bola/s': {'grade': 'SM_1-1/8"_h', 'type': 'Bola/s', 'product_name': 'Cotton'},
             'Ivory Coast Manbo/s': {'grade': 'SM_1-1/8"_h', 'type': 'Manbo/s', 'product_name': 'Cotton'},
             'Mali Juli/s': {'grade': 'SM_1-1/8"_h', 'type': 'Juli/s', 'product_name': 'Cotton'},
             'Memphis/Orleans/Texas': {'grade': 'M_1-1/8"_std', 'type': 'MOT', 'product_name': 'Cotton'},
             'A Index': {'grade': 'M_1-1/8"_std', 'type': 'A Index', 'product_name': 'Cotton'}}

wood_dict = {'France_DKD_Sapele': {'origin': 'France', 'grade': 'Sapele', 'type': 'DKD', 'instrument_id': 4342,
                                   'product_name': 'wood'},
             'Netherlands_FKD_Sapelli': {'origin': 'Netherlands', 'grade': 'Sapelli', 'type': 'FKD',
                                         'instrument_id': 4341, 'product_name': 'Wood'}}


class PhysicalContractRefLoader(ContractRefLoader):

    def __init__(self, instrument_name, source, params=None):
        super().__init__(instrument_name=instrument_name, source=source, params=params)
        self.instrument_name = instrument_name
        self.grade = self.params.get("grade")
        self.origin = self.params.get("origin")
        self.type = self.params.get("type")
        self.crop_year_type = None
        self.data_source = self.params.get("data_source") #, "cotlook")
        self.instrument_id = self.params.get("instrument_id")
        self.product_name = self.params.get("product_name")

    def determine_crop_year_type(self, crop_year: str):
        self.crop_year_type = 'cross' if '/' in crop_year else 'straight'

    def load_ref_data(self) -> pd.DataFrame:
        product_name_filter = f"p.name = '{self.product_name}'"
        if self.product_name == 'Cotton':
            if self.instrument_name == 'A Index':
                external_name_filter = "pc.external_name LIKE '%A Index'"  # match variants like '2024/2025 A Index'
            else:
                external_name_filter = f"pc.external_name = '{self.instrument_name}'"
        elif self.product_name == 'Wood':
            pass
        query = f"""
                    SELECT DISTINCT pc.external_name, pi.origin, pi.type, pi.grade, pi.crop_year
                    FROM ref.physical_contract pc 
                    JOIN market.physical_market_price pmp ON pc.traded_contract_id = pmp.traded_contract_id
                    JOIN ref.physical_instrument pi ON pi.id = pc.instrument_id
                    WHERE pmp.data_source = '{self.data_source}'
                      AND {external_name_filter}
                      AND {product_name_filter}
                """

        if self.grade:
            query += f" AND pi.grade = '{self.grade}'"
        if self.origin:
            query += f" AND pi.origin = '{self.origin}'"
        if self.type:
            query += f" AND pi.type = '{self.type}'"

        query += " LIMIT 5"

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        if not df.empty:
            row = df.iloc[0]
            self.origin = row['origin']
            self.grade = row['grade']
            self.type = row['type']
            self.determine_crop_year_type(row['crop_year'])

        return df

    def load_contracts(self):
        query = f"""
            SELECT pc.external_name, pc.shipment_month, pi.grade, pi.origin, pmp.tdate
            FROM ref.physical_contract pc 
            JOIN market.physical_market_price pmp 
            ON pc.traded_contract_id = pmp.traded_contract_id
            JOIN ref.physical_instrument pi 
            ON pi.id = pc.instrument_id
            WHERE pc.external_name = '{self.instrument_name}'
        """
        if self.grade:
            query += f" AND pi.grade = '{self.grade}'"
        if self.origin:
            query += f" AND pi.origin = '{self.origin}'"

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(query), conn)

        df['tdate'] = pd.to_datetime(df['tdate'], errors='coerce')
        return df

    def load_expiry_dates(self, start_date, end_date):
        """Load list of expiry dates for active contracts for the contract_ref_loader within a date range."""
        pass

    def load_shipment_month(self):
        pass
