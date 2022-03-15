## coding: UTF-8
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

def convert_kerasmodel_to_frozen_pb(model, pbmodelname, im_preprocess=None):
    input_hwc    = model.inputs[0].shape[1:]
    input_name   = 'e2pose/inputimg'
    output_names = [layer.name for layer in model.outputs]
    pb_model     = tf.function(lambda x: model(x, training=False)) if im_preprocess is None else tf.function(lambda x: model(im_preprocess(x), training=False))
    pb_model     = pb_model.get_concrete_function(tf.TensorSpec(shape=[None,]+input_hwc, dtype=tf.float32, name=input_name))
    pb_model_frozen = convert_variables_to_constants_v2(pb_model)
    pb_model_frozen.graph.as_graph_def()
    logdir, name = os.path.split(pbmodelname)
    tf.io.write_graph(graph_or_graph_def=pb_model_frozen.graph, logdir=logdir, name=name, as_text=False)
    config = {'input_name':input_name, 'output_names':output_names}
    with open(os.path.join(logdir, 'config.json'), 'w') as fid:
        json.dump(config, fid)

def convert_savedmodel_to_trtmodel(src_path, dst_path, input_kwd='inputimg', precision_mode='FP16', max_batch_size=1, max_workspace_size_bytes=1 << 20, minimum_segment_size=3, maximum_cached_engines=int(1e3), build=False):
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=precision_mode,
                                                                   max_workspace_size_bytes=max_workspace_size_bytes,
                                                                   maximum_cached_engines=maximum_cached_engines,
                                                                   use_calibration=False,
                                                                   allow_build_at_runtime=True)
    converter = trt.TrtGraphConverterV2(input_saved_model_dir=str(src_path), use_dynamic_shape=False, conversion_params=conversion_params)
    converter.convert()
    if build:
        _model = tf.saved_model.load(str(src_path)).signatures["serving_default"]
        _bhwc  = [1,]+[_v for _v in _model.inputs[0].shape[1:]]
        del _model
        def tmp_input_fn():
            yield (np.random.normal(size=_bhwc).astype(np.float32),)
        converter.build(input_fn=tmp_input_fn)
    converter.save(output_saved_model_dir=str(dst_path))

# Ref: https://stackoverflow.com/questions/44329185/convert-a-graph-proto-pb-pbtxt-to-a-savedmodel-for-use-in-tensorflow-serving-o/44329200#44329200
def convert_frozen_to_savedmodel(graph_path, saved_path, input_kwd='inputimg', output_kwds=['pv/concat', 'kvxy/concat'], name='e2pose'):
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants
    builder = tf.compat.v1.saved_model.Builder(str(saved_path))
    with tf.io.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    sigs = {}
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        g = tf.compat.v1.get_default_graph()
        tf.import_graph_def(graph_def, name=name)
        input_op       = [_op for _op in g.get_operations() if input_kwd in _op.name][0]
        output_ops     = [[_op for _op in g.get_operations() if _kwd in _op.name][-1] for _kwd in output_kwds]
        input_tensor   = {input_kwd:input_op.outputs[0]}
        output_tensors = {_kwd:_op.outputs[0] for _kwd,_op in zip(output_kwds, output_ops)}

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.compat.v1.saved_model.predict_signature_def(input_tensor, output_tensors)
        builder.add_meta_graph_and_variables(sess,
                                            [tag_constants.SERVING],
                                            signature_def_map=sigs)
    builder.save()
    
