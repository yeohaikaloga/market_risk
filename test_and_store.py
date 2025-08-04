if __name__ == '__main__':
    uat_engine = get_engine('uat')
    with uat_engine.connect() as connection:
        Gin_S6_query = 'SELECT * FROM staging.cotton_shankar6_price'
        Gin_S6 = pd.read_sql_query(text(Gin_S6_query), con=connection)
        print(Gin_S6.head())

if __name__ == '__main__':
    prod_engine = get_engine('prod')
    with prod_engine.connect() as connection:
        test_query = """SELECT dc.traded_contract_id,
        dc.contract_ref_loader,
        mp.tdate,
        mp.px_settle_last_dt,
        mp.px_settle
        FROM ref.derivatives_contract dc
        JOIN market.market_price mp
        ON dc.traded_contract_id = mp.traded_contract_id
        WHERE dc.contract_ref_loader LIKE 'CT%%'
        AND LENGTH(dc.contract_ref_loader) = 4
        AND dc.futures_category = 'Fibers'
        AND SUBSTRING(dc.contract_ref_loader FROM 3 FOR 1) IN ('H', 'K', 'N', 'Z')"""
        test = pd.read_sql_query(text(test_query), con=connection)
        print(test.head())