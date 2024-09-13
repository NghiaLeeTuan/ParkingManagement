import oracledb

class Connection:
    def __init__(self):
        try:
            # Kết nối với cơ sở dữ liệu
            self.con = oracledb.connect(user='sys', password='123', dsn='localhost:1521/orcl', mode=oracledb.SYSDBA)
            self.cursor = self.con.cursor() 
            print("Connection successful")
        except oracledb.DatabaseError as e:
            print("There is a problem with Oracle:", e)
            self.con = None
            self.cursor = None

    def Execute(self, query):
        if not self.cursor:
            raise Exception("Not connected to the database")
        try:
            # Thực thi câu lệnh SQL
            self.cursor.execute(query)
            result = self.cursor.fetchall()               
            return result

        except oracledb.DatabaseError as e:
            error, = e.args
            return f"Database error: {error.message}"

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.con:
            self.con.close()