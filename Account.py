import connection

con = connection.Connection()
query = ''

def Check_Account(Account, Password):
    query = "SELECT Pass_acc FROM accounts WHERE ID_acc = '"+Account+"'"
    result = con.Execute(query)
    for row in result:
        if Password == row[0]:
            return True
    return False