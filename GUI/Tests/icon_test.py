#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.startUI()


    def startUI(self):
        self.setGeometry(100, 200, 300, 400)
        self.setWindowTitle('Mycon')
        self.setWindowIcon(QIcon('cool_icon.jpg'))
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
