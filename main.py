import connection 
from PyQt6.QtWidgets import QApplication, QMainWindow

import Form_Account, Form_Parking
import sys


ui = ""
app = QApplication(sys.argv)
MainWindow= QMainWindow()
con = connection.Connection()
def Login():
    global ui
    ui = Form_Account.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    


Login()
sys.exit(app.exec())


