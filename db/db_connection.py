from sqlalchemy import create_engine
from config import UAT_DB_PARAMS, PROD_DB_PARAMS, RMS_DB_PARAMS
from urllib.parse import quote_plus

def get_engine(db):
    params_map = {
        'uat': {**UAT_DB_PARAMS, 'dialect': 'postgresql+psycopg2'},
        'prod': {**PROD_DB_PARAMS, 'dialect': 'postgresql+psycopg2'},
        'rms': {**RMS_DB_PARAMS, 'dialect': 'mysql+pymysql'}
    }

    if db not in params_map:
        raise ValueError(f"Unknown db '{db}', expected one of {list(params_map.keys())}")

    params = params_map[db]

    user_enc = quote_plus(params['user'])
    password_enc = quote_plus(params['password'])

    db_url = f"{params['dialect']}://{user_enc}:{password_enc}@{params['host']}:{params['port']}/{params['database']}"

    connect_args = {}
    if db == 'rms':
        # Optional: If you run into plugin/auth issues, uncomment this:
        connect_args = {
            "ssl": {
                "ssl_ca": "/path/to/ca.pem",
                "ssl_cert": "/path/to/client-cert.pem",
                "ssl_key": "/path/to/client-key.pem"
            }
        }

        # Optional SSL args:
        # connect_args = {"ssl": {"ssl_ca": "/path/to/ca.pem"}}
        return create_engine(db_url, connect_args=connect_args)

    return create_engine(db_url)
