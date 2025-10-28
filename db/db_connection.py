from sqlalchemy import create_engine
from config import UAT_DB_PARAMS, PROD_DB_PARAMS, DZRMS_DB_PARAMS
from urllib.parse import quote_plus


def get_engine(db):
    params_map = {
        'uat': UAT_DB_PARAMS,
        'prod': PROD_DB_PARAMS,
        'dzrms': DZRMS_DB_PARAMS
    }

    if db not in params_map:
        raise ValueError(f"Unknown db '{db}', expected one of {list(params_map.keys())}")

    params = params_map[db]

    user_enc = quote_plus(params['user'])
    password_enc = quote_plus(params['password'])

    if db == 'dzrms':
        # MySQL
        dialect_driver = "mysql+pymysql"
    else:
        # PostgreSQL
        dialect_driver = "postgresql+psycopg2"

    driver = 'mysql+pymysql' if db == 'dzrms' else 'postgresql'
    db_url = f"{driver}://{user_enc}:{password_enc}@{params['host']}:{params['port']}/{params['database']}"
    print(f"Connecting to: {db_url}")

    return create_engine(db_url)
