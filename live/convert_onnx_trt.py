## coding: UTF-8
import os
import sys
import argparse
import pathlib
import tensorflow as tf
import tempfile

sys.path += [os.path.dirname(__file__), os.path.dirname(os.path.dirname(__file__))]

from utils.define import POSE_DATASETS


from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from collections import OrderedDict
import subprocess

#--------------
# Parse args
#--------------
def parse_args():
    from distutils.util import strtobool
    parser = argparse.ArgumentParser()
    parser.add_argument('--freezed_model', type=pathlib.Path)
    parser.add_argument('--onnx_model'   , type=pathlib.Path, default=None)
    parser.add_argument('--trt_model'    , type=pathlib.Path, default=None)
    args = parser.parse_args()
    return args


#--------------
# Main
#--------------
if __name__ == "__main__":
    args = parse_args()
    print(args)

    with tf.io.gfile.GFile(args.freezed_model, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    tmp_saved_model = os.path.join(tempfile.gettempdir(), 'saved_model')
    builder = tf.compat.v1.saved_model.Builder(tmp_saved_model)
    
    sigs = {}
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        graph = tf.compat.v1.get_default_graph()
        tf.import_graph_def(graph_def, name='e2pose')
        input_op    = [op for op in graph.get_operations() if 'inputimg' in op.name][0]
        out_pv_ops  = [op for op in graph.get_operations() if '_pv/Sigmoid' in op.name]
        out_kv_ops  = [op for op in graph.get_operations() if '_kv/Sigmoid' in op.name]
        out_kyx_ops = [op for op in graph.get_operations() if '_kyx_offset' in op.name]
        if len(out_kyx_ops) < 0:
            out_kyx_ops = [op for op in graph.get_operations() if '_kyx_scaled' in op.name]
        if len(out_kyx_ops) < 0:
            out_kyx_ops = [op for op in graph.get_operations() if '_kyx_reshape' in op.name]
        
        input_tensor   = {'input':input_op.outputs[0]}
        output_tensors = OrderedDict()
        for _ii, _op in enumerate(out_pv_ops):
            output_tensors[f'out_pv/{_ii}'] = _op.outputs[0]
        for _ii, _op in enumerate(out_kv_ops):
            output_tensors[f'out_kv/{_ii}'] = _op.outputs[0]
        for _ii, _op in enumerate(out_kyx_ops):
            output_tensors[f'out_yx/{_ii}'] = _op.outputs[0]
        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.compat.v1.saved_model.predict_signature_def(input_tensor, output_tensors)
        builder.add_meta_graph_and_variables(sess,
                                            [tag_constants.SERVING],
                                            signature_def_map=sigs)
    builder.save()
    
    cmd = f'python3 -m tf2onnx.convert --opset 13 --saved-model {tmp_saved_model} --output {args.onnx_model}'
    subprocess.run(cmd.split())
    
    cmd = f'trtexec --onnx={args.onnx_model} --explicitBatch --saveEngine={args.trt_model} --fp16 --verbose'
    subprocess.run(cmd.split())