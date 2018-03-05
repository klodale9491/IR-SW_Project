import mysql.connector

class DBConnector:
    @staticmethod
    def connect():
        mysql_config = {
            'user': 'root',
            'password': '',
            'host': '127.0.0.1',
            'database': 'giallo_zafferano',
            'raise_on_warnings': True,
        }
        cnx = mysql.connector.connect(**mysql_config, buffered=True)
        return cnx
