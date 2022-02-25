## coding: UTF-8
import sys
import os
import cv2
import numpy as np
import matplotlib.cm as cm

sys.path += [os.path.dirname(__file__)]
from define import POSE_DATASETS
from e2pose_io import check_is_movie, check_is_image, read_src, check_fps, seaquence_writer

class Painter():
    def __init__(self, dataset_name, thickness=0.003, draw_joint=True, draw_limbs=True, th=0.5):
        self.dataset      = POSE_DATASETS[dataset_name]
        self.thickness    = thickness
        self.draw_joint   = draw_joint
        self.draw_limbs   = draw_limbs
        self.th           = th
        self.colors_joint = self.dataset['colors_joint'] if 'colors_joint' in self.dataset else self.create_colors(self.dataset['joints'])
        self.colors_limbs = self.dataset['colors_limbs'] if 'colors_limbs' in self.dataset else self.create_colors(self.dataset['skeleton'])
        
    
    def create_colors(self, items):
        return [(np.array(cm.hsv(ii/(len(items)-1))[:3])*255).astype(np.uint8).tolist() for ii in range(len(items))]

    def __call__(self, image, anns):
        thickness = max(1, int(np.sqrt(np.sum(np.square(image.shape[:2]))) * self.thickness))
        # Draw limbs
        if self.draw_limbs:
            for person in anns:
                kpts = np.reshape(person['keypoints'], [-1,3])
                for ii, (idx1, idx2) in enumerate(self.dataset['skeleton']):
                    color  = self.colors_limbs[ii]
                    j1, j2 = kpts[idx1], kpts[idx2]
                    if (j1[-1] > self.th) and (j2[-1] > self.th):
                        cv2.line(image, tuple(j1[:2].astype(np.int32).tolist()), tuple(j2[:2].astype(np.int32).tolist()), tuple(color), thickness=thickness, lineType=cv2.LINE_AA)
        # Draw Joint
        if self.draw_joint:
            for person in anns:
                kpts = np.reshape(person['keypoints'], [-1,3])
                for ii, j1 in enumerate(kpts):
                    color  = self.colors_joint[ii]
                    if (j1[-1] > self.th):
                        cv2.circle(image, tuple(j1[:2].astype(np.int32).tolist()), radius=thickness, color=tuple(color), thickness=-1, lineType=cv2.LINE_AA)
        return image
    
    def add_fps_text(self, image, fps):
        thickness  = max(1, int(np.sqrt(np.sum(np.square(image.shape[:2]))) * self.thickness))
        text       = '{:.2f} FPS'.format(fps)
        fontHeight = image.shape[0] / 20
        fontScale  = fontHeight / 24
        org        = (10, 10 + int(fontHeight))
        color      = (np.clip(np.array(cm.cool(1/fps)[:3]), 0, 1)*255).astype(np.uint8).tolist()
        return cv2.putText(image, text=text,
                            org=org,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=fontScale,
                            color=color,
                            thickness=thickness,
                            lineType=cv2.LINE_AA)

def rescale_kpts(pred, src_hw, dst_hw):
    ratio_h = dst_hw[0] / src_hw[0]
    ratio_w = dst_hw[1] / src_hw[1]
    for ii in range(len(pred)):
        kpts       = np.reshape(pred[ii]['keypoints'], [-1,3])
        kpts[:,0] *= ratio_w
        kpts[:,1] *= ratio_h
        pred[ii]['keypoints'] = np.reshape(kpts,[-1]).tolist()
    return pred