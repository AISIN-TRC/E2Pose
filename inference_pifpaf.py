## coding: UTF-8
import os, tqdm, argparse, pathlib, json, glob, cv2, time, shutil
import numpy as np

from utils import draw
from utils.define import POSE_DATASETS

import openpifpaf

#--------------
# Parse args
#--------------
def parse_args():
    from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt'           , type=str         , default='shufflenetv2k30')
    parser.add_argument('--dst'            , type=pathlib.Path, default=pathlib.Path('./outsample'))
    parser.add_argument('--src'            , type=str         , default='none')
    parser.add_argument('--dataset'        , type=str         , default='COCO')
    parser.add_argument('--input_wh'       , type=str         , default='512,512')
    parser.add_argument('--dev'            , type=strtobool   , default=False)
    parser.add_argument('--add_fps'        , type=strtobool   , default=True)
    parser.add_argument('--add_blur'       , type=strtobool   , default=True)
    args          = parser.parse_args()
    args.src      = glob.glob(args.src)
    args.dev      = bool(args.dev)
    args.add_fps  = bool(args.add_fps)
    args.add_blur = bool(args.add_blur)
    args.input_wh = [int(_v) for _v in args.input_wh.split(',')]
    return args


#--------------
# Main
#--------------
if __name__ == "__main__":
    args = parse_args()
    print(args)

    openpifpaf.network.Factory(checkpoint=args.ckpt, download_progress=True).factory()
    model   = openpifpaf.predictor.Predictor(checkpoint=args.ckpt)
    painter = draw.Painter(args.dataset, th=0.01)
    
    with draw.seaquence_writer(args.dst, dev=args.dev) as writer:
        for ii, (src_path, raw, frame) in tqdm.tqdm(enumerate(draw.read_src(args.src, resize=args.input_wh))):
            stt           = time.perf_counter()
            pred, _, meta = model.numpy_image(frame)
            pred          = [ann.json_data() for ann in pred]
            pred_time     = time.perf_counter() - stt
            humans        = draw.rescale_kpts(pred, frame.shape[:2], raw.shape[:2])
            if args.add_blur:
                _size = [int(_v/70) for _v in raw.shape[:2][::-1]]
                raw   = cv2.blur(raw, _size)
            image     = painter(raw, humans)
            if args.add_fps:
                image = painter.add_fps_text(image, 1/pred_time)
            writer(src_path, image, {'time':pred_time, 'humans':humans})
    