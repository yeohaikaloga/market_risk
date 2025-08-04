from position_loader.position_loader import PositionLoader
import pandas as pd
from sqlalchemy import text

class DerivativesPositionLoader(PositionLoader):

    def __init__(self, date, source):
        super().__init__(source)
        self.source = source
        self.date = date

    def load_position(self, date, product, trader_id=None, counterparty_id=None) -> pd.DataFrame:

        ## TO IMPLEMENT ADDITIONAL JOIN TO REGION/BOOK HIERARCHY TABLE FOR COTTON

        product_map = {
            'cotton': ['cto'],
            'rms-cfs': ['cfs'],
            'rms': ['cfs', 'rmc'],
            'rubber': ['rba', 'rbc']
            # Add others like 'rubber', 'wood', etc. as needed
        }

        if product not in product_map:
            raise ValueError(f"Unsupported product: {product}")

        opera_products = product_map[product]

        # Core SELECT and GROUP BY columns
        select_cols = [
            "pos.tdate", "sp.subportfolio", "pf.portfolio",
            "sec.security_id", "sec.strike", "sec.derivative_type", "sec.currency",
            "SUM(pos.total_active_lots) AS total_active_lots"
        ]
        group_by_cols = select_cols[:-1]  # everything except the aggregate

        joins = [
            "JOIN position_opera.sub_portfolio sp ON pos.sub_portfolio_id = sp.id",
            "JOIN position_opera.portfolio pf ON sp.portfolio_id = pf.id",
            "JOIN position_opera.security sec ON pos.risk_security_id = sec.id"
        ]
        where_conditions = [
            "pos.opera_product IN :opera_products",
            "pos.tdate = :date",
            "sp.subportfolio != 'CONSO-CT'"
        ]
        params = {
            "opera_products": tuple(opera_products),  # tuple for SQLAlchemy IN clause
            "date": date
        }

        # Optional trader
        if trader_id is not None:
            joins.append("JOIN position_opera.trader tr ON pos.trader_id = tr.id")
            select_cols += ["tr.id AS trader_id", "tr.name AS trader_name"]
            group_by_cols += ["tr.id", "tr.name"]
            where_conditions.append("tr.id = :trader_id")
            params["trader_id"] = trader_id

        # Optional counterparty
        if counterparty_id is not None:
            joins.append("JOIN position_opera.counterparty cp ON pos.counterparty_id = cp.id")
            select_cols += ["cp.id AS counterparty_id", "cp.counterparty_parent"]
            group_by_cols += ["cp.id", "cp.counterparty_parent"]
            where_conditions.append("cp.id = :counterparty_id")
            params["counterparty_id"] = counterparty_id

        # Build query
        base_query = f"""
            SELECT {', '.join(select_cols)}
            FROM position_opera.position pos
            {' '.join(joins)}
            WHERE {' AND '.join(where_conditions)}
            GROUP BY {', '.join(group_by_cols)}
        """

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(base_query), conn, params=params)

        return df

        def load_position_ref_data():
            pass

