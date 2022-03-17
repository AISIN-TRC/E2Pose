## coding: UTF-8
import os
import sys
import json
import traceback
import numpy as np

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QGraphicsView, QFrame, QGraphicsScene, QDockWidget, QPushButton, QGraphicsItem, QComboBox, QSlider
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QPainter

sys.path += [os.path.dirname(__file__), os.path.dirname(os.path.dirname(__file__))]
from utils.define import POSE_DATASETS
from style import *
from model import E2PoseThread, MODEL_PATH
from camera import CameraCapture

# --- DEFINEs ---
QT_PALETTE_UI_JSON = './live/config/pallete_pref.json'

class PoseGraph(QGraphicsItem):
    def __init__(self, human, dataset, parent=None, name='human', size=5.0, th=0.5):
        super().__init__(parent)
        self.setToolTip(name)
        self.add_circles(human, dataset, size, th)

    def add_circles(self, human, dataset, size=5.0, th=0.5):
        kpts        = np.reshape(human['keypoints'], [-1,3])
        self.rect   = {'top':np.inf,'bottom':-np.inf,'left':np.inf, 'right':-np.inf}
        self.joints = []
        for j1, pen in zip(kpts, dataset['pen_joint']):
            if (j1[-1] > th):
                top    = j1[0]-size/2
                left   = j1[1]-size/2
                bottom = top + size
                right  = left + size
                self.joints.append({'pen': pen, 'top':int(top), 'left':int(left), 'size':int(size)})
                self.rect['top']    = min(self.rect['top'], top)
                self.rect['left']   = min(self.rect['left'], left)
                self.rect['bottom'] = max(self.rect['bottom'], bottom)
                self.rect['right']  = min(self.rect['right'], right)
        
        self.limbs = []
        for (idx1, idx2), pen in zip(dataset['skeleton'], dataset['pen_limbs']):
            j1, j2 = kpts[idx1], kpts[idx2]
            if (j1[-1] > th) and (j2[-1] > th):
                self.limbs.append({'pen': pen, 'x1':int(j1[0]), 'y1':int(j1[1]), 'x2':int(j2[0]), 'y2':int(j2[1])})

    
    def update_item(self, index):
        self.index = index
        self.update()

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        for joint in self.joints:
            painter.setPen(joint['pen'])
            painter.drawEllipse(joint['top'], joint['left'], joint['size'], joint['size'])
        for limb in self.limbs:
            painter.setPen(limb['pen'])
            painter.drawLine(limb['x1'], limb['y1'], limb['x2'], limb['y2'])
            

    def boundingRect(self):
        return QRectF(self.rect['top'], self.rect['left'], self.rect['bottom']-self.rect['top'], self.rect['right']-self.rect['left'])

class E2PoseDock(QDockWidget):
    def __init__(self, *args, contextMenu=None, view=None, dataset='COCO', **kwargs):
        super().__init__(*args, **kwargs)
        self.view  = view
        self.scene = view.scene
        self.annot = None
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.setFeatures(QDockWidget.DockWidgetMovable|QDockWidget.DockWidgetFloatable)
        self.setStyleSheet(DOCK_STYLE)
        self.create_layout()
        self.finish_model()
        self.set_dataset(dataset)
        self.create_model()
    
    def set_dataset(self, dataset):
        self.dataset = POSE_DATASETS[dataset]
        colors_joint = self.dataset['colors_joint'] if 'colors_joint' in self.dataset else self.create_colors(self.dataset['joints'])
        self.dataset['pen_joint'] = [QPen(QColor(*_color)) for _color in colors_joint]
        [_pen.setWidth(1) for _pen in self.dataset['pen_joint']]
        colors_limbs = self.dataset['colors_limbs'] if 'colors_limbs' in self.dataset else self.create_colors(self.dataset['skeleton'])
        self.dataset['pen_limbs'] = [QPen(QColor(*_color)) for _color in colors_limbs]
        [_pen.setWidth(1) for _pen in self.dataset['pen_limbs']]

    def create_colors(self, items):
        import matplotlib.cm as cm
        return [(np.array(cm.hsv(ii/(len(items)-1))[:3])*255).astype(np.uint8).tolist() for ii in range(len(items))]

    def create_model(self):
        self.model  = E2PoseThread()
        self.camera = CameraCapture()
        self.model.sig_ret_result.connect(self.draw_result)
        self.model.sig_ret_result.connect(self.camera.get_next)
        self.camera.sig_frame.connect(self.model.predict_frame)
        self.camera.finished.connect(self.stop_predict)
    
    def change_threshold(self, val):
        self.model.set_threshold(val/100.0)

    def create_layout(self):
        self.widget = QWidget()
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)
        self.setWidget(self.widget)

        self.create_model_selector()
        self.slider_th = QSlider(Qt.Horizontal)
        self.slider_th.setMinimum(0)
        self.slider_th.setMaximum(100)
        self.slider_th.setValue(50)
        self.slider_th.setStyleSheet(SLIDER_STYLE)
        self.slider_th.valueChanged.connect(self.change_threshold)
        self.btn_start = QPushButton('Start')
        self.btn_stop  = QPushButton('Stop')
        self.btn_start.setStyleSheet(BUTTON_STYLE)
        self.btn_stop.setStyleSheet(BUTTON_STYLE)
        self.btn_start.clicked.connect(self.start_predict)
        self.btn_stop.clicked.connect(self.stop_predict)
        self.layout.addWidget(self.slider_th)
        self.layout.addWidget(self.btn_start)
        self.layout.addWidget(self.btn_stop)

    def create_model_selector(self):
        self.combo = QComboBox()
        self.combo.setStyleSheet(COMBO_STYLE)
        self.combo.addItems(sorted(list(MODEL_PATH.keys())))
        self.layout.addWidget(self.combo)
    
    def start_model(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
    def finish_model(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def draw_humans(self, raw_rgb, humans):
        del self.annot
        pix        = QPixmap.fromImage(QImage(raw_rgb, raw_rgb.shape[1], raw_rgb.shape[0], QImage.Format_RGB888))
        self.annot = [PoseGraph(human, dataset=self.dataset, name='human') for human in humans]
        [self.view.scene.removeItem(item) for item in self.view.scene.items() if item.toolTip()=='human']
        self.view.setPixmap(pix)
        [self.view.scene.addItem(item) for item in self.annot]

    def draw_result(self, result):
        self.draw_humans(result['img'], result['humans'])

    def start_predict(self, *args, **kwargs):
        self.model.set_model_path(self.combo.currentText())
        self.start_model()
        self.camera.start()
        self.model.start()

    def stop_predict(self, *args, **kwargs):
        self.camera.stop()
        self.model.stop()
        self.camera.wait()
        self.model.wait()
        self.finish_model()

class LiveView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(WINDOW_STYLE)
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.m_pixmap_item = self.scene.addPixmap(QPixmap())
    
    def setPixmap(self, pixmap):
        init_pixmap = self.m_pixmap_item.pixmap().isNull()
        self.m_pixmap_item.setPixmap(pixmap)
        if init_pixmap:
            self.fitInView(self.m_pixmap_item, QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        if not self.m_pixmap_item.pixmap().isNull():
            self.fitInView(self.m_pixmap_item, QtCore.Qt.KeepAspectRatio)
        return super().resizeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setWindowTitle('E2Pose live')
        
        # --- Main widgets
        self.cwidget = QWidget()
        self.clayout = QVBoxLayout()
        self.clayout.setContentsMargins(0, 0, 0, 0)
        self.cwidget.setLayout(self.clayout)
        self.setCentralWidget(self.cwidget)

        # --- Create canvas
        self.live  = LiveView(self)
        self.clayout.addWidget(self.live)

        # --- Load model
        self.model = E2PoseDock('Model', self, view=self.live)
        self.addDockWidget(Qt.RightDockWidgetArea, self.model) # Set to dock area
        
        # --- Last ---
        self.openWindow()


    # WindowOpen
    def openWindow(self):
        try:
            with open(QT_PALETTE_UI_JSON) as f:
                pref = json.load(f)
            try:
                info = pref['MainWindow']
                self.setPosition(info['x'], info['y'], info['w'], info['h'])
            except:
                traceback.print_exc()
        except:
            traceback.print_exc()

    # WindowClose
    def closeWindow(self):
        print('[INFO] Save Window Layout')
        pref = {}
        x,y,w,h = self.getPosition()
        pref['MainWindow'] = {'x':x, 'y':y, 'w':w, 'h':h}
        with open(QT_PALETTE_UI_JSON, 'w') as f:
            json.dump(pref, f, indent=4)
    
    def getPosition(self):
        return self.geometry().x(), self.geometry().y(), self.geometry().width(), self.geometry().height()
    
    def setPosition(self, px, py, pw, ph):
        self.setGeometry(px, py, pw, ph)

#--------------
# Main
#--------------
if __name__ == "__main__":
    pass