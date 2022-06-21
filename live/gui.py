## coding: UTF-8
import os
import glob
import sys
import json
import numpy as np
import cv2

from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QFrame, QGraphicsScene, QDockWidget, QPushButton, QGraphicsItem, QComboBox, QSlider, QGroupBox, QFileDialog, QLabel, QCheckBox, QLineEdit
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QPainter

sys.path += [os.path.dirname(__file__), os.path.dirname(os.path.dirname(__file__))]
from utils.define import POSE_DATASETS
from style import *
from model import E2PoseThread
from camera import CameraCapture

from logging import getLogger
logger = getLogger(__name__)

# --- DEFINEs ---
QT_USER_CONFIG = './live/config/pallete_pref.json'

class PoseGraph(QGraphicsItem):
    def __init__(self, human, dataset, parent=None, name='human', circle_size=5.0, th=0.5):
        super().__init__(parent)
        self.setToolTip(name)
        self.draw_skeleton(human, dataset, circle_size, th)

    def draw_skeleton(self, human, dataset, circle_size=5.0, th=0.5):
        kpts        = np.reshape(human['keypoints'], [-1,3])
        self.rect   = {'top':np.inf,'bottom':-np.inf,'left':np.inf, 'right':-np.inf}
        self.joints = []
        for j1, pen in zip(kpts, dataset['pen_joint']):
            if (j1[-1] > th):
                top    = j1[0]-circle_size/2
                left   = j1[1]-circle_size/2
                bottom = top + circle_size
                right  = left + circle_size
                self.joints.append({'pen': pen, 'top':int(top), 'left':int(left), 'size':int(circle_size)})
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
        self.view    = view
        self.scene   = view.scene
        self.annot   = None
        self.dst_mov = None
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

    def create_threshold_slider(self):
        group = QGroupBox('threshold:')
        vbox  = QVBoxLayout()
        group.setLayout(vbox)
        self.slider_th = QSlider(Qt.Horizontal)
        self.slider_th.setMinimum(0)
        self.slider_th.setMaximum(100)
        self.slider_th.setValue(50)
        self.slider_th.setStyleSheet(SLIDER_STYLE)
        self.slider_th.valueChanged.connect(self.change_threshold)
        vbox.addWidget(self.slider_th)
        self.layout.addWidget(group)
    
    def change_line_width(self, val):
        [_pen.setWidth(val) for _pen in self.dataset['pen_joint']]
        [_pen.setWidth(val) for _pen in self.dataset['pen_limbs']]
    
    def change_bg_color(self, color_str):
        color = QColor(color_str)
        if color.isValid():
            self.view.setBackgroundBrush(color)

    def create_draw_config_palette(self):
        group = QGroupBox('drawing settings:')
        vbox  = QVBoxLayout()
        group.setLayout(vbox)
        widget = QWidget()
        hbox1  = QHBoxLayout()
        widget.setLayout(hbox1)
        vbox.addWidget(widget)
        hbox1.addWidget(QLabel('circle size:'))
        self.slider_circle = QSlider(Qt.Horizontal)
        self.slider_circle.setMinimum(3)
        self.slider_circle.setMaximum(31)
        self.slider_circle.setValue(5)
        self.slider_circle.setTickInterval(2)
        hbox1.addWidget(self.slider_circle)
        widget = QWidget()
        hbox2  = QHBoxLayout()
        widget.setLayout(hbox2)
        vbox.addWidget(widget)
        hbox2.addWidget(QLabel('line width:'))
        self.slider_linewidth = QSlider(Qt.Horizontal)
        self.slider_linewidth.setMinimum(1)
        self.slider_linewidth.setMaximum(10)
        self.slider_linewidth.setValue(1)
        self.slider_linewidth.valueChanged.connect(self.change_line_width)
        hbox2.addWidget(self.slider_linewidth)

        widget = QWidget()
        hbox3  = QHBoxLayout()
        widget.setLayout(hbox3)
        vbox.addWidget(widget)
        hbox3.addWidget(QLabel('background color:'))
        self.bg_color_edit = QLineEdit("#000000")
        self.bg_color_edit.textEdited.connect(self.change_bg_color)
        hbox3.addWidget(self.bg_color_edit)

        self.btn_hide_image = QCheckBox('Hide image')
        self.btn_export_mp4 = QCheckBox('Export mp4')
        self.btn_export_mp4.stateChanged.connect(self.export_mp4_changed)
        vbox.addWidget(self.btn_hide_image)
        vbox.addWidget(self.btn_export_mp4)

        self.layout.addWidget(group)

    def create_start_stop_button(self):
        self.btn_start = QPushButton('Start')
        self.btn_stop  = QPushButton('Stop')
        self.btn_save  = QPushButton('Save')
        self.btn_start.setStyleSheet(BUTTON_STYLE)
        self.btn_stop.setStyleSheet(BUTTON_STYLE)
        self.btn_save.setStyleSheet(BUTTON_STYLE)
        self.btn_start.clicked.connect(self.start_predict)
        self.btn_stop.clicked.connect(self.stop_predict)
        self.btn_save.clicked.connect(self.export_image)
        self.layout.addWidget(self.btn_stop)
        self.layout.addWidget(self.btn_start)
        self.layout.addWidget(self.btn_save)

    def create_device_selector(self):
        group = QGroupBox('input device:')
        vbox  = QVBoxLayout()
        group.setLayout(vbox)
        self.device = QComboBox()
        self.device.setStyleSheet(COMBO_STYLE)
        self.device.addItems(glob.glob('/dev/video*') + ['movie file'])
        self.device.activated.connect(self.change_device)
        vbox.addWidget(self.device)
        self.layout.addWidget(group)

    def change_device(self, idx):
        txt = self.device.itemText(idx)
        if txt == 'movie file':
            fname = QFileDialog.getOpenFileName(self, 'Select movie file', './', 'Movie file (*.avi *.mp4 *.mov *.wmv *.mpg)')
            if fname[0]:
                self.device.setItemText(idx, fname[0])
                self.device.addItem('movie file')

    def create_layout(self):
        self.widget = QWidget()
        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout)
        self.setWidget(self.widget)
        self.create_device_selector()
        self.create_model_selector()
        self.create_threshold_slider()
        self.create_draw_config_palette()
        self.layout.addStretch()
        self.create_start_stop_button()

    def create_model_selector(self):
        group = QGroupBox('inference model')
        vbox  = QVBoxLayout()
        group.setLayout(vbox)
        self.combo = QComboBox()
        self.combo.setStyleSheet(COMBO_STYLE)
        self.combo.addItems(glob.glob('./pretrains/**/*.trt', recursive=True))
        vbox.addWidget(self.combo)
        self.layout.addWidget(group)
    
    def start_model(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
    def finish_model(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_export_mp4.setChecked(0)

    def draw_humans(self, raw_rgb, humans):
        del self.annot
        h,w,c = raw_rgb.shape
        alpha = np.ones_like(raw_rgb[:,:,0])*255
        alpha = alpha.reshape((h,w,1))
        if self.btn_hide_image.isChecked():
            alpha = alpha * 0
        raw_rgba   = np.concatenate([raw_rgb,alpha], axis=-1)
        pix        = QPixmap.fromImage(QImage(raw_rgba, raw_rgba.shape[1], raw_rgba.shape[0], QImage.Format_RGBA8888))
        self.annot = [PoseGraph(human, dataset=self.dataset, name='human', circle_size=self.slider_circle.value()) for human in humans]
        [self.view.scene.removeItem(item) for item in self.view.scene.items() if item.toolTip()=='human']
        self.view.setPixmap(pix)
        [self.view.scene.addItem(item) for item in self.annot]
        self.write_dst_mov()

    def draw_result(self, result):
        self.draw_humans(result['img'], result['humans'])

    def start_predict(self, *args, **kwargs):
        self.model.set_model_path(self.combo.currentText())
        self.camera.set_device(self.device.currentText())
        self.start_model()
        self.camera.start()
        self.model.start()

    def stop_predict(self, *args, **kwargs):
        self.camera.stop()
        self.model.stop()
        self.camera.wait()
        self.model.wait()
        self.finish_model()
    
    def export_image(self, *args, **kwargs):
        fileName = QFileDialog.getSaveFileName(self, "Export image File", "./sample_out/untitled.png", "Images (*.png *.xpm *.jpg)")[0]
        if len(fileName)>3:
            logger.info(str('Export ') + str(fileName))
            pixMap = self.view.grab()
            pixMap.save(fileName)
            
    def write_dst_mov(self):
        if self.dst_mov is not None:
            cv_image = qimage_to_cv(self.view.grab().toImage())
            if (self.dst_mov['video'] is None) or (self.dst_mov['wh'] is None):
                self.dst_mov['wh'] = cv_image.shape[:2][::-1]
                self.dst_mov['video'] = cv2.VideoWriter(self.dst_mov['path'],
                                                        self.dst_mov['codec'],
                                                        30, tuple(self.dst_mov['wh']))
            cv_image = cv2.resize(cv_image[:,:,:3], self.dst_mov['wh'])
            self.dst_mov['video'].write(cv_image)

    def export_mp4_changed(self, state):
        if state > 0:
            fileName = QFileDialog.getSaveFileName(self, "Export movie File", "./sample_out/untitled.avi", "Movies (*.avi *.mp4)")[0]
            if len(fileName)>3:
                _ext = fileName.split('.')[-1].lower()
                if _ext == 'wmv':
                    _codec = 'H263'
                elif _ext == 'mp4':
                    _codec = 'MP4V'
                else:
                    _codec = 'XVID'

                logger.info(str('Export ') + str(fileName))
                self.dst_mov = {
                    'video': None,
                    'wh': None,
                    'codec': cv2.VideoWriter_fourcc(*_codec),
                    'path': fileName,
                }
        else:
            if self.dst_mov is not None:
                if self.dst_mov['video'] is not None:
                    self.dst_mov['video'].release()
                self.dst_mov = None


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
        if os.path.isfile(QT_USER_CONFIG):
            with open(QT_USER_CONFIG) as f:
                pref = json.load(f)
            if 'MainWindow' in pref:
                self.setPosition(pref['MainWindow']['x'], pref['MainWindow']['y'], pref['MainWindow']['w'], pref['MainWindow']['h'])
            if 'threshold' in pref:
                self.model.slider_th.setValue(pref['threshold'])
            if 'circle_size' in pref:
                self.model.slider_circle.setValue(pref['circle_size'])
            if 'line_width' in pref:
                self.model.slider_linewidth.setValue(pref['line_width'])


    # WindowClose
    def closeWindow(self):
        logger.info('[INFO] Save config')
        x,y,w,h = self.getPosition()
        pref    = {
                    'MainWindow'  : {'x':x, 'y':y, 'w':w, 'h':h},
                    'threshold'   : self.model.slider_th.value(),
                    'circle_size' : self.model.slider_circle.value(),
                    'line_width'  : self.model.slider_linewidth.value(),
                  }
        if not os.path.isdir(os.path.dirname(QT_USER_CONFIG)):
            os.makedirs(os.path.dirname(QT_USER_CONFIG))
        with open(QT_USER_CONFIG, 'w') as f:
            json.dump(pref, f, indent=4)
        if self.model.dst_mov is not None:
            if self.model.dst_mov['video'] is not None:
                self.model.dst_mov['video'].release()
            self.model.dst_mov = None
    
    def getPosition(self):
        return self.geometry().x(), self.geometry().y(), self.geometry().width(), self.geometry().height()
    
    def setPosition(self, px, py, pw, ph):
        self.setGeometry(px, py, pw, ph)

def qimage_to_cv(qimage):
    w, h, d = qimage.size().width(), qimage.size().height(), qimage.depth()
    bytes_ = qimage.bits().asstring(w * h * d // 8)
    arr = np.frombuffer(bytes_, dtype=np.uint8).reshape((h, w, d // 8))
    return arr

#--------------
# Main
#--------------
if __name__ == "__main__":
    pass