import mysql.connector
from config import MYSQL_CONFIG
import pandas as pd

_connection = None

def get_connection():
    global _connection
    if _connection is None or not _connection.is_connected():
        _connection = mysql.connector.connect(**MYSQL_CONFIG)
    return _connection

def get_index(index_name, timestamp):
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT value FROM market_indices WHERE index_name = %s AND timestamp <= %s ORDER BY timestamp DESC LIMIT 1"
    cursor.execute(query, (index_name, timestamp))
    row = cursor.fetchone()
    cursor.close()
    return row[0] if row else 0.0
