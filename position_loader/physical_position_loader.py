from position.position_loader import PositionLoader
import pandas as pd
from sqlalchemy import text

class PhysicalPositionLoader(PositionLoader):

    def __init__(self, date, source):
        super().__init__(source)
        self.source = source
        self.date = date

    def load_position(self, date, opera_product, trader_id=None, counterparty_id=None) -> pd.DataFrame:
        base_query = '''
                SELECT pos.cob_date, sp.subportfolio, pf.portfolio, sec.security_id, sec.strike, sec.derivative_type,
                       tr.id AS trader_id, tr.name AS trader_name,
                       cp.id AS counterparty_id, cp.counterparty_parent,
                       SUM(pos.total_active_lots) AS total_active_lots
                FROM position_opera.position_loader pos
                JOIN position_opera.sub_portfolio sp ON pos.sub_portfolio_id = sp.id
                JOIN position_opera.portfolio pf ON sp.portfolio_id = pf.id
                JOIN position_opera.counterparty cp ON pos.counterparty_id = cp.id
                JOIN position_opera.trader tr ON pos.trader_id = tr.id
                JOIN position_opera.security sec ON pos.risk_security_id = sec.id
                WHERE pos.opera_product = :opera_product
                  AND pos.cob_date = :date
                  AND sp.subportfolio != 'CONSO-CT'
            '''

        # Prepare parameters dictionary
        params = {
            "opera_product": opera_product,
            "date": date,
        }

        # Add optional filters
        if trader_id is not None:
            base_query += " AND tr.id = :trader_id"
            params["trader_id"] = trader_id

        if counterparty_id is not None:
            base_query += " AND cp.id = :counterparty_id"
            params["counterparty_id"] = counterparty_id

        # Add GROUP BY clause
        base_query += '''
                GROUP BY pos.cob_date, sp.subportfolio, pf.portfolio, sec.security_id, sec.strike, sec.derivative_type,
                         tr.id, tr.name, cp.id, cp.counterparty_parent
            '''

        with self.source.connect() as conn:
            df = pd.read_sql_query(text(base_query), conn, params=params)

        return df

