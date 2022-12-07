## coding: UTF-8
import os
import tqdm
import cv2
import json
import glob
import shutil
import pathlib

def check_is_movie(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in ['.mov','.avi','.mp4','.mpeg', '.mov']

def check_is_image(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in ['.bmp','.png','.jpeg','.jpg']


def read_src(*args, resize=None):
    def _resize(frame):
        if (resize is not None):
            return cv2.resize(frame, tuple(resize[::-1]))
        else:
            return frame
    def _read_file(src):
        if check_is_movie(src):
            cap = cv2.VideoCapture(src)
            while 1:
                ret, raw = cap.read()
                if not ret:
                    break
                raw  = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                yield src, raw, _resize(raw)
            cap.release()
        elif check_is_image(src):
            raw  = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
            yield src, raw, _resize(raw)
            
    src_list = sum(args, [])
    for src in src_list:
        for path, raw, frame in _read_file(src):
            yield path, raw, frame

def check_fps(mov):
    cap = cv2.VideoCapture(mov)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

class seaquence_writer():
    def __init__(self, dst_dir, dev=False):
        self.dst_dir   = dst_dir if isinstance(dst_dir, pathlib.Path) else pathlib.Path(dst_dir)
        self.dev       = dev
        self.latest    = None
        self.devdata   = []
        self.frame_idx = 0

    def write(self, src_path):
        if check_is_image(src_path):
            self._write(src_path)
        elif check_is_movie(src_path):
            if self.latest != src_path:
                self._write(self.latest, src_path, is_mov=True)
            else:
                self._write(src_path, is_mov=True)
            self.latest = src_path

    def __enter__(self):
        return self
    
    def get_mov_frame_name(self, src_path, frame_idx=None):
        dst_path = self.dst_dir / os.path.splitext(os.path.basename(src_path))[0]
        dst_path = dst_path / ('%09d.jpg' % self.frame_idx if frame_idx is not None else '*.jpg')
        return dst_path

    def close_dev_data(self):
        if self.latest is not None:
            if self.dev:
                dev_json_path = str(self.dst_dir / (os.path.basename(self.latest) + '.json'))
                with open(dev_json_path, 'w') as f:
                    json.dump(self.devdata, f, indent=4)
            self.devdata = []
    
    def export_video(self, dst_path, files, fps):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video  = None
        for file in tqdm.tqdm(files, desc=str(dst_path), ncols=120):
            frame = cv2.imread(file)
            if video is None:
                video = cv2.VideoWriter(dst_path, fourcc, fps, frame.shape[:2][::-1])
            video.write(frame)
        if video is not None:
            video.release()

    def close_movie_data(self):
        self.frame_idx = 0
        if self.latest is not None:
            if check_is_movie(self.latest):
                dst_mov = str(self.dst_dir / (os.path.basename(self.latest) + '.mp4'))
                fps     = check_fps(self.latest)
                files   = sorted(glob.glob(str(self.get_mov_frame_name(self.latest))))
                self.export_video(dst_mov, files, fps)
                if not self.dev:
                    shutil.rmtree(os.path.dirname(files[0]))

    def close_data(self):
        self.close_movie_data()
        if self.dev:
            self.close_dev_data()
        self.latest = None

    def write_image(self, src_path, image):
        if check_is_image(src_path):
            dst_path = self.dst_dir / os.path.basename(src_path)
        elif check_is_movie(src_path):
            dst_path = self.get_mov_frame_name(src_path, self.frame_idx)
            self.frame_idx += 1
        dst_path.parent.mkdir(mode=0o777, parents=True, exist_ok=True)
        cv2.imwrite(str(dst_path), image[:,:,::-1])

    def write_dev(self, src_path, image, data):
        if src_path != self.latest:
            self.close_data()
            self.latest = src_path
        self.devdata.append(data)
        self.write_image(src_path, image)

    def __call__(self, src_path, image, data={}):
        if src_path != self.latest:
            self.close_data()
        self.write_dev(src_path, image, data)

    def __exit__(self, *args, **kwargs):
        self.close_data()