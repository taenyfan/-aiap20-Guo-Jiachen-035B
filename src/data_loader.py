import sqlite3
import pandas as pd

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    table_name = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'][0]
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df
