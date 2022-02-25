## coding: UTF-8
import os, tqdm, argparse, pathlib, json, glob, cv2, time, shutil
import numpy as np

from utils import draw, MMPOSE_DEFINE
from utils.define import POSE_DATASETS

import mmcv
from mmpose.apis import init_pose_model, process_mmdet_results, inference_top_down_pose_model
from mmdet.apis import inference_detector, init_detector

MMDET_CONFIGS  = {os.path.splitext(os.path.basename(_conf))[0]:{'conf':_conf, 'ckpt':_ckpt} for _conf,_ckpt in MMPOSE_DEFINE.DETECTORS}
MMPOSE_CONFIGS = {os.path.splitext(os.path.basename(_conf))[0]:{'conf':_conf, 'ckpt':_ckpt} for _conf,_ckpt in MMPOSE_DEFINE.TOPDOWN_POSEMODELS}

#--------------
# Parse args
#--------------
def parse_args():
    from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('--det'            , type=str         , default='yolov3_d53_320_273e_coco')
    parser.add_argument('--pose'           , type=str         , default='shufflenetv2_coco_256x192')
    parser.add_argument('--dst'            , type=pathlib.Path, default=pathlib.Path('./outsample'))
    parser.add_argument('--src'            , type=str         , default='none')
    parser.add_argument('--dataset'        , type=str         , default='COCO')
    parser.add_argument('--dev'            , type=strtobool   , default=False)
    parser.add_argument('--add_fps'        , type=strtobool   , default=True)
    parser.add_argument('--add_blur'       , type=strtobool   , default=True)
    args          = parser.parse_args()
    args.src      = glob.glob(args.src)
    args.dev      = bool(args.dev)
    args.add_fps  = bool(args.add_fps)
    args.add_blur = bool(args.add_blur)
    return args

def mmcv_file_read(src_path):
    if draw.check_is_movie(src_path):
        video_reader = mmcv.VideoReader(src_path)
        for frame in mmcv.track_iter_progress(video_reader):
            yield frame
    elif draw.check_is_image(src_path):
        pass
    return []

#--------------
# Main
#--------------
if __name__ == "__main__":
    args = parse_args()
    print(args)

    mmdet_config  = MMDET_CONFIGS[args.det]
    mmpose_config = MMPOSE_CONFIGS[args.pose]
    det_model    = init_detector(mmdet_config['conf'], mmdet_config['ckpt'])
    pose_model   = init_pose_model(mmpose_config['conf'], mmpose_config['ckpt'])
    dataset      = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    
    if not hasattr(dataset_info, 'flip_pairs'):
        from collections import namedtuple
        dataset_name = 'TopDownCocoDataset'
        flip_pairs   = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        dataset_info = namedtuple('DatasetInfo', ['dataset_name', 'flip_pairs'])(dataset_name, flip_pairs)

    painter = draw.Painter(args.dataset, th=0.3)
    with draw.seaquence_writer(args.dst, dev=args.dev) as writer:
        for ii, (src_path, raw, _) in tqdm.tqdm(enumerate(draw.read_src(args.src))):
            frame          = raw[:,:,::-1].astype(np.uint8)
            stt            = time.perf_counter()
            mmdet_results  = inference_detector(det_model, frame)
            person_results = process_mmdet_results(mmdet_results, 1)
            _humans, _     = inference_top_down_pose_model(pose_model, frame, person_results, bbox_thr=0.3, dataset=dataset, dataset_info=dataset_info, format='xyxy')
            pred_time      = time.perf_counter() - stt
            humans         = [{'keypoints':np.reshape(human['keypoints'], [-1]).tolist()} for human in _humans if ('keypoints' in human)]
            if args.add_blur:
                _size = [int(_v/70) for _v in raw.shape[:2][::-1]]
                raw   = cv2.blur(raw, _size)
            image     = painter(raw, humans)
            if args.add_fps:
                image = painter.add_fps_text(image, 1/pred_time)
            writer(src_path, image, {'time':pred_time, 'humans':humans})
