## coding: UTF-8
import pathlib
import cv2
from queue import Queue
from PyQt5.QtCore import pyqtSignal, QThread

from logging import getLogger
logger = getLogger(__name__)

class CameraCapture(QThread):
    sig_frame = pyqtSignal(object)

    def __init__(self, src=None):
        super().__init__()
        if not pathlib.Path(str(src)).is_file():
            src = 0
        self.src  = src
        self.next = Queue()

    def set_device(self, src):
        self.src  = src

    def get_next(self, *args, **kwargs):
        self.next.put(True)

    def stop(self, *args, **kwargs):
        if self.isRunning():
            logger.info('Stop capture')
            self.next.put(False)

    def run(self):
        self.cap = cv2.VideoCapture(self.src)
        while 1:
            ret, raw = self.cap.read()
            if not ret:
                logger.info('Close capture')
                break
            raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            self.sig_frame.emit(raw_rgb)
            if not self.next.get():
                break
        self.cap.release()

#--------------
# Main
#--------------
if __name__ == "__main__":
    pass

