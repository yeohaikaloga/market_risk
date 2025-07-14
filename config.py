import configparser

config = configparser.ConfigParser()
config.read('C:/Vault/config.ini')

def load_db_params(section):
    return {
        'host': config.get(section, 'host'),
        'port': config.get(section, 'port'),
        'database': config.get(section, 'database'),
        'user': config.get(section, 'user'),
        'password': config.get(section, 'password'),
    }

UAT_DB_PARAMS = load_db_params('uat')
PROD_DB_PARAMS = load_db_params('prod')