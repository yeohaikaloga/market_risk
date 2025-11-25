import mysql.connector
import pandas as pd
import datetime

def connect_to_database():
    # Connect to the database using mysql-connector-python
    conn = mysql.connector.connect(

        host='SGTCX-OPRDB01',
        database='ro_rmsvar',
        user='zhihui',
        password='5dqfAWR#K1Noj9#1Q'
    )
    return conn

def read_from_database(table_name, sql_query=""):
    """
    Connects to a database, executes an SQL query, and returns headers and data.
    """
    conn = None

    try:
        # Assuming you have already established a database connection (conn)
        conn = connect_to_database()  # Replace with your actual connection setup

        if not sql_query:
            # If query is empty, return the entire table
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        else:
            # Execute the provided SQL query
            df = pd.read_sql_query(sql_query, conn)

        headers = df.columns.tolist()

    except Exception as e:
        print(f"Error: {e}")
        headers = []
        df = pd.DataFrame()

    finally:
        if conn:
            conn.close()  # Close the connection if it's open

    return headers, df
    #return headers, df.iloc[:, 1:] # Exclude the first column (headers)
    #return headers, df.iloc[1:]  # Exclude the first row (headers)

def fetch_data_from_database(SqlQuery):

    try:
        conn = connect_to_database()  # Replace with your actual connection setup
        cursor = conn.cursor()

        # Execute the SQL query
        cursor.execute(SqlQuery)

        # Fetch all rows
        PriceChangeVector = cursor.fetchall()

        # Reverse the order of rows (if needed)
        #PriceChangeVector = np.flipud(PriceChangeVector)

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return PriceChangeVector

    except Exception as e:
        print(f"Error fetching value_date: {e}")
        return None

def get_value_date():
    # Connect to the database
    conn = connect_to_database()
    cursor = conn.cursor()

    cursor.execute("SELECT value_date FROM valuation_date_table_view")

    results = cursor.fetchall()
    # Convert the results to a list of values
    DBValueDate = results[0][0]
    print("DBValueDate:", DBValueDate)
    print("type:", type(DBValueDate))

    return DBValueDate
    #return DBValueDate.strftime('%Y-%m-%d')

def get_settle_date():
    # Connect to the database
    conn = connect_to_database()

    cursor = conn.cursor()

    cursor.execute("SELECT settlement_date FROM valuation_date_table_view")

    results = cursor.fetchall()
    # Convert the results to a list of values
    DBSettleDate = results[0][0]
    #print("DBValueDate:", DBSettleDate)
    #print("type:", type(DBSettleDate))

    return DBSettleDate

def getXLSFilename_buname_with_path(ReportName, InBUName, folder_path):
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"{ReportName}_{InBUName}_{current_timestamp}.xlsx"
    file_path = f"{folder_path}\\{report_name}"
    return file_path


if __name__ == '__main__':
# Example usage
    table_name = "helper_var_mult_factor_rmsvar_view"
    sql_query = "SELECT * FROM helper_var_mult_factor_rmsvar_view WHERE value_date = '2024-07-01'"
    sql_query1 = "SELECT * FROM helper_var_accounting_report_view_cfs WHERE value_date = '2024-07-02'"
    headers, data = read_from_database(table_name, sql_query1)

    if headers:
        print("Headers:", headers)
        print("Data:")
        print(data)
    else:
        print("Error fetching data from the database.")

    ValueDate = get_value_date()
    print("DBValueDate:", ValueDate)

    tablename2 = "var_aggregation_level_table_cfs_view"
    sql_query2 = "select distinct(aggregation_level) from var_aggregation_level_table_cfs_view "
    InAggregationLevel = fetch_data_from_database(sql_query2)
    print("InAggregationLevel:")
    print(InAggregationLevel)

    file_path = 'C:\\Users\\ding.zhihui\\OneDrive - Olam International\\Desktop\\VaR Project (FY24)\\Opera RMS VaR\\VaR Results'
    InBUName = 'cfs'
    # Example usage:
    try:
        report_name_with_path = getXLSFilename_buname_with_path("Test", InBUName, file_path)
        print(f"Generated report name with path: {report_name_with_path}")
    except Exception as e:
        print(f"Error: {e}")

