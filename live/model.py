## coding: UTF-8
import os
import pathlib
import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
TRT_LOGGER = trt.Logger()

from queue import Queue
from PyQt5.QtCore import pyqtSignal, QThread

from logging import getLogger
logger = getLogger(__name__)

# --- DEFINEs ---
QT_PALETTE_UI_JSON = './live/config/pallete_pref.json'
#MODEL_PATH = {
#    'coco_res101_512': os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pretrains/COCO/ResNet101/512x512/model.trt'),
#}

# Ref: https://github.com/sa-kei728/pytorch-YOLOv4/blob/master/demo_trt.py
# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def get_engine(engine_path):
    # If a serialized engine exists, use it instead of building an engine.
    logger.info("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
                
def allocate_buffers(engine, batch_size):
    inputs   = None
    outputs  = {}
    bindings = []
    stream   = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)
        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            logger.info("Binding [" + str(binding) + "] : Input")
            if inputs is not None:
                raise Exception('Not support yet')
            inputs = {'mem':HostDeviceMem(host_mem, device_mem), 'shape':dims, 'dtype':dtype }
        else:
            logger.info("Binding [" + str(binding) + "] : Output")
            outputs[str(binding)] = {'mem':HostDeviceMem(host_mem, device_mem), 'shape':dims, 'dtype':dtype }
    return inputs, outputs, bindings, stream
    
class E2PoseThread(QThread):
    sig_get_frame  = pyqtSignal(object)
    sig_ret_result = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.th_pv      = 0.5
        self.raw_rgb    = Queue()
        self.model_path = None
        
    def set_threshold(self, th):
        self.th_pv = th

    @staticmethod
    def convert_coco_style(p_kv, p_yx, im_hw):
        im_hw =  np.reshape(im_hw, [ 1,2])
        p_kv  =  np.reshape(p_kv , [-1,1])
        p_xy  = (np.reshape(p_yx , [-1,2]) * im_hw)[:,::-1]
        p_xyv =  np.concatenate([p_xy,p_kv], axis=-1)
        return {'keypoints': np.reshape(p_xyv, [-1]).tolist(), 'category_id':1}

    def decode_trt(self, trt_out, im_hw):
        ret_humans = []
        n_map      = len(trt_out) // 3
        for ii_map in range(n_map):
            pv = np.array(trt_out[f'out_pv/{ii_map}'])
            kv = np.array(trt_out[f'out_kv/{ii_map}'])
            yx = np.array(trt_out[f'out_yx/{ii_map}'])
            p_mask = np.squeeze(pv >= self.th_pv, axis=-1)
            for p_kv, p_yx in zip(kv[p_mask], yx[p_mask]):
                human = self.convert_coco_style(p_kv, p_yx, im_hw)
                ret_humans.append(human)
        return ret_humans
    
    def predict_frame(self, raw_rgb):
        self.raw_rgb.put(raw_rgb)
    
    def set_model_path(self, model_path=None):
        if pathlib.Path(str(model_path)).exists():
            self.model_path = pathlib.Path(str(model_path))
    
    def stop(self):
        if self.isRunning():
            logger.info('Stop model')
            self.raw_rgb.put(None)

    def run(self):
        if not pathlib.Path(str(self.model_path)).exists():
            raise Exception('Model file is not found')
            
        logger.info('Start model')
        cuda.init()
        dev = cuda.Device(0)
        ctx = dev.make_context()

        with get_engine(self.model_path) as engine, engine.create_execution_context() as context:
            inputs, outputs, bindings, stream = allocate_buffers(engine, 1)

            def predict_trt(raw_rgb):
                data = cv2.resize(raw_rgb, inputs['shape'][1:3][::-1]).astype(inputs['dtype'])
                inputs['mem'].host = np.ascontiguousarray(np.stack([data], axis=0))
                cuda.memcpy_htod_async(inputs['mem'].device, inputs['mem'].host, stream)
                context.execute_async(bindings=bindings, stream_handle=stream.handle)
                [cuda.memcpy_dtoh_async(out['mem'].host, out['mem'].device, stream) for key, out in outputs.items()]
                stream.synchronize()
                return {key:np.reshape(out['mem'].host, out['shape']) for key, out in outputs.items()}
                
            while 1:
                raw_rgb = self.raw_rgb.get()
                if raw_rgb is None:
                    break
                trt_out = predict_trt(raw_rgb)
                humans  = self.decode_trt(trt_out, im_hw=raw_rgb.shape[:2])
                self.sig_ret_result.emit({'img':raw_rgb, 'humans':humans})

        ctx.pop()
        del ctx

    

#--------------
# Main
#--------------
if __name__ == "__main__":
    pass