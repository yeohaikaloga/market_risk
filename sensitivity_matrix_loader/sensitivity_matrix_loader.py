import pandas as pd
from utils.log_utils import get_logger


logger = get_logger(__name__)
class SensitivityMatrixLoader:

    def __init__(self, cob_date: str, product: str, source):
        self.cob_date = cob_date
        self.product = product
        self.source = source
        pass

    def load_sensitivity_matrix(self) -> pd.DataFrame:

        """
        Loads the sensitivity matrix for a specific cob_date and product.
        """
        def execute_and_extract(conn, query, column_name):

            # Execute the query only once
            temp_df = pd.read_sql(query, conn)

            if temp_df.empty:
                return None, 0  # Return None for the string and 0 for count

            # The result is guaranteed to have at least one row here
            # Safely extract the JSON string from the first row (index 0)
            json_string = temp_df[column_name].iloc[0]

            return json_string, len(temp_df)

        with self.source.connect() as conn:
            if self.product == 'cotton':
                cto_query = f"""
                    SELECT cotton_sensitivity_report  
                    FROM staging.opera_sensitivity_reports 
                    WHERE staging.opera_sensitivity_reports.cob_date = '{self.cob_date}'
                    """
                json_string, count = execute_and_extract(conn, cto_query, 'cotton_sensitivity_report')
                if json_string:
                    logger.info(f"Found sensitivity matrix for {self.product} on {self.cob_date}.")
                    df = pd.read_json(json_string)
                else:
                    message = f"ERROR: No sensitivity report found for {self.product} on {self.cob_date}."
                    logger.warning(message)
                    print(message)

            elif self.product == 'rubber':
                df_rba = pd.DataFrame()
                df_rbc = pd.DataFrame()

                rba_query = f"""
                    SELECT non_china_rubber_sensitivity_report 
                    FROM staging.opera_sensitivity_reports 
                    WHERE staging.opera_sensitivity_reports.cob_date = '{self.cob_date}'
                    """
                json_string_rba, count_rba = execute_and_extract(conn, rba_query, 'non_china_rubber_sensitivity_report')

                if json_string_rba:
                    logger.info(f"Found sensitivity matrix for Non-China rubber on {self.cob_date}.")
                    df_rba = pd.read_json(json_string_rba)
                else:
                    logger.warning(f"No sensitivity report found for Non-China rubber on {self.cob_date}.")

                rbc_query = f"""
                    SELECT china_rubber_sensitivity_report 
                    FROM staging.opera_sensitivity_reports 
                    WHERE staging.opera_sensitivity_reports.cob_date = '{self.cob_date}
                    '"""

                json_string_rbc, count_rbc = execute_and_extract(conn, rbc_query, 'china_rubber_sensitivity_report')
                if json_string_rbc:
                    logger.info(f"Found sensitivity matrix for China rubber on {self.cob_date}.")
                    df_rbc = pd.read_json(json_string_rbc)
                else:
                    logger.warning(f"No sensitivity report found for China rubber on {self.cob_date}.")

                df = pd.concat([df_rba, df_rbc], ignore_index=True)

            df['match_key'] = (df['subportfolio'].astype(str) + '|' +
                               df['strike'].astype(str) + '|' +
                               df['product_code'].astype(str) + '|' +
                               df['contract_month'].astype(str) + '|' +
                               df['derivative_type'].astype(str) + '|' +
                               df['active_lots'].astype(str)
                               )
            logger.info(f"Total rows in {self.product} sensitivity matrix: {len(df)}")
            return df

        return pd.DataFrame()
