## coding: UTF-8
import tensorflow as tf

class E2PoseModel(tf.keras.models.Model):
    def __init__(self, inputs, outputs, name=None, **kwargs):
        super().__init__(inputs, outputs, name=name)
    
    @staticmethod
    def get_preprosess(backbone):
        if backbone.startswith('MobileNetV3'):
            print('Use MobileNetV3 preprocess')
            ret_fnc = tf.keras.applications.mobilenet_v3.preprocess_input
        elif backbone.startswith('MobileNetV2'):
            print('Use MobileNetV2 preprocess')
            ret_fnc = tf.keras.applications.mobilenet_v2.preprocess_input
        elif backbone.startswith('MobileNet'):
            print('Use MobileNet preprocess')
            ret_fnc = tf.keras.applications.mobilenet.preprocess_input
        elif backbone.startswith('ResNet50V2'):
            print('Use ResNet50V2 preprocess')
            ret_fnc = tf.keras.applications.resnet_v2.preprocess_input
        elif backbone.startswith('ResNet50'):
            print('Use ResNet50 preprocess')
            ret_fnc = tf.keras.applications.resnet50.preprocess_input
        elif backbone.startswith('NASNet'):
            print('Use NASNet preprocess')
            ret_fnc = tf.keras.applications.nasnet.preprocess_input
        elif backbone.startswith('EfficientNetB'):
            print('Use EfficientNetB preprocess')
            ret_fnc = tf.keras.applications.efficientnet.preprocess_input
        elif backbone.startswith('EfficientNetV2'):
            print('Use EfficientNetV2 preprocess')
            ret_fnc = tf.keras.applications.efficientnet_v2.preprocess_input
        else:
            print('Use Std preprocess')
            ret_fnc = lambda x : (x / 127.0) - 1.0
        return lambda x : ret_fnc(tf.cast(x, tf.float32))