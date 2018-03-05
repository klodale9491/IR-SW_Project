import mysql.connector

class DBConnector:
    @staticmethod
    def connect(user, pwd, host, db):
        mysql_config = {
            'user': user,
            'password': pwd,
            'host': host,
            'database': db,
            'raise_on_warnings': True,
        }
        cnx = mysql.connector.connect(**mysql_config, buffered=True)
        return cnx
