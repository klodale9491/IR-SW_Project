import mysql.connector

class DBConnector:
    @staticmethod
    def connect(db="recipes_irsw"):
        mysql_config = {
            'user': 'root',
            'password': 'root',
            'host': '127.0.0.1',
            'database': db,
            'raise_on_warnings': True,
        }
        cnx = mysql.connector.connect(**mysql_config, buffered=True)
        return cnx