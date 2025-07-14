from sqlalchemy import create_engine, inspect, text
from config import UAT_DB_PARAMS, PROD_DB_PARAMS
from urllib.parse import quote_plus


def get_engine(db):
    params_map = {
        'uat': UAT_DB_PARAMS,
        'prod': PROD_DB_PARAMS,
    }

    if db not in params_map:
        raise ValueError(f"Unknown db '{db}', expected one of {list(params_map.keys())}")

    params = params_map[db]

    user_enc = quote_plus(params['user'])
    password_enc = quote_plus(params['password'])

    db_url = (
        f"postgresql://{user_enc}:{password_enc}@"
        f"{params['host']}:{params['port']}/{params['database']}"
    )

    return create_engine(db_url)

