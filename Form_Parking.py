# Form implementation generated from reading ui file '.\UI\Parking.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(918, 418)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.scrollArea = QtWidgets.QScrollArea(parent=self.centralwidget)
        self.scrollArea.setGeometry(QtCore.QRect(80, 40, 241, 241))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 239, 239))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.img_in = QtWidgets.QLineEdit(parent=self.scrollAreaWidgetContents)
        self.img_in.setGeometry(QtCore.QRect(10, 10, 221, 221))
        self.img_in.setObjectName("img_in")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.btn_checkin = QtWidgets.QPushButton(parent=self.centralwidget)
        self.btn_checkin.setGeometry(QtCore.QRect(150, 340, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_checkin.setFont(font)
        self.btn_checkin.setObjectName("btn_checkin")
        self.btn_checkout = QtWidgets.QPushButton(parent=self.centralwidget)
        self.btn_checkout.setGeometry(QtCore.QRect(630, 340, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_checkout.setFont(font)
        self.btn_checkout.setObjectName("btn_checkout")
        self.widget = QtWidgets.QWidget(parent=self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(670, 70, 120, 80))
        self.widget.setObjectName("widget")
        self.scrollArea_2 = QtWidgets.QScrollArea(parent=self.centralwidget)
        self.scrollArea_2.setGeometry(QtCore.QRect(560, 40, 241, 241))
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 239, 239))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.img_out = QtWidgets.QLineEdit(parent=self.scrollAreaWidgetContents_2)
        self.img_out.setGeometry(QtCore.QRect(10, 10, 221, 221))
        self.img_out.setObjectName("img_out")
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.txt_number = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.txt_number.setGeometry(QtCore.QRect(340, 90, 201, 31))
        self.txt_number.setObjectName("txt_number")
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(190, 0, 31, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(660, 0, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.txt_time_in = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.txt_time_in.setGeometry(QtCore.QRect(100, 300, 201, 31))
        self.txt_time_in.setObjectName("txt_time_in")
        self.label_3 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(410, 70, 61, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 310, 61, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(490, 310, 71, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.txt_time_out = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.txt_time_out.setGeometry(QtCore.QRect(580, 300, 201, 31))
        self.txt_time_out.setObjectName("txt_time_out")
        self.txt_fee = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.txt_fee.setGeometry(QtCore.QRect(340, 150, 201, 31))
        self.txt_fee.setObjectName("txt_fee")
        self.label_6 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(430, 130, 31, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.txt_stt = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.txt_stt.setGeometry(QtCore.QRect(340, 210, 201, 31))
        self.txt_stt.setObjectName("txt_stt")
        self.label_7 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(410, 190, 61, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.btn_check = QtWidgets.QPushButton(parent=self.centralwidget)
        self.btn_check.setGeometry(QtCore.QRect(450, 250, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_check.setFont(font)
        self.btn_check.setObjectName("btn_check")
        self.btn_edit = QtWidgets.QPushButton(parent=self.centralwidget)
        self.btn_edit.setGeometry(QtCore.QRect(330, 250, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_edit.setFont(font)
        self.btn_edit.setObjectName("btn_edit")
        self.btn_logout = QtWidgets.QPushButton(parent=self.centralwidget)
        self.btn_logout.setGeometry(QtCore.QRect(810, 10, 101, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_logout.setFont(font)
        self.btn_logout.setObjectName("btn_logout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 918, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_checkin.setText(_translate("MainWindow", "Check In"))
        self.btn_checkout.setText(_translate("MainWindow", "Check Out"))
        self.label.setText(_translate("MainWindow", "IN"))
        self.label_2.setText(_translate("MainWindow", "OUT"))
        self.label_3.setText(_translate("MainWindow", "NUMBER"))
        self.label_4.setText(_translate("MainWindow", "TIME IN"))
        self.label_5.setText(_translate("MainWindow", "TIME OUT"))
        self.label_6.setText(_translate("MainWindow", "FEE"))
        self.label_7.setText(_translate("MainWindow", "STATUS"))
        self.btn_check.setText(_translate("MainWindow", "Check"))
        self.btn_edit.setText(_translate("MainWindow", "Edit"))
        self.btn_logout.setText(_translate("MainWindow", "Log out"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
