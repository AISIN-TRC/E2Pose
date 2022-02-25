## coding: UTF-8
import os
import json
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

def convert_kerasmodel_to_frozen_pb(model, pbmodelname, im_preprocess=None):
    input_hwc    = model.inputs[0].shape[1:]
    input_name   = 'e2pose/inputimg'
    output_names = [layer.name for layer in model.outputs]
    pb_model     = tf.function(lambda x: model(x)) if im_preprocess is None else tf.function(lambda x: model(im_preprocess(x)))
    pb_model     = pb_model.get_concrete_function(tf.TensorSpec(shape=[None,]+input_hwc, dtype=tf.float32, name=input_name))
    pb_model_frozen = convert_variables_to_constants_v2(pb_model)
    pb_model_frozen.graph.as_graph_def()
    logdir, name = os.path.split(pbmodelname)
    tf.io.write_graph(graph_or_graph_def=pb_model_frozen.graph, logdir=logdir, name=name, as_text=False)
    config = {'input_name':input_name, 'output_names':output_names}
    with open(os.path.join(logdir, 'config.json'), 'w') as fid:
        json.dump(config, fid)

def convert_savedmodel_to_trtmodel(src_path, dst_path, precision_mode='FP16', max_batch_size=1, max_workspace_size_bytes=1 << 20, minimum_segment_size=3, maximum_cached_engines=int(1e3)):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode='FP16',
                                                                   max_workspace_size_bytes=max_workspace_size_bytes,
                                                                   maximum_cached_engines=maximum_cached_engines,
                                                                   use_calibration=False,
                                                                   allow_build_at_runtime=True)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=str(src_path), use_dynamic_shape=False, conversion_params=conversion_params)
    converter.convert()
    converter.save(output_saved_model_dir=str(dst_path))