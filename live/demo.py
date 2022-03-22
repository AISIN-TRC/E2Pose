## coding: UTF-8
import os
import sys
import argparse

from PyQt5.QtWidgets import QApplication

sys.path += [os.path.dirname(__file__), os.path.dirname(os.path.dirname(__file__))]
from gui import MainWindow

from logging import getLogger
logger = getLogger(__name__)

#--------------
# Parse args
#--------------
def parse_args():
    from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

#--------------
# Main
#--------------
if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    app      = QApplication(sys.argv)
    main_gui = MainWindow(app)
    main_gui.show()
    app.exec_()
    main_gui.closeWindow()
    sys.exit()

