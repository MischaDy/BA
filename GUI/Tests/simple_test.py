#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QWidget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QWidget()
    w.resize(500, 250)
    w.move(0, 0)
    w.setWindowTitle('Simpleyy')
    w.show()

    sys.exit(app.exec_())
