## coding: UTF-8
import tensorflow as tf


class VariableDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        if not isinstance(rate, tf.Variable):
            self.rate = tf.Variable(initial_value=rate, dtype=tf.float32, trainable=False)

    def get_config(self):
        base_config = super().get_config()
        base_config['rate'] = base_config['rate'] if not isinstance(base_config['rate'], tf.Variable) else float(base_config['rate'].numpy())
        return base_config
    
# Ref. http://musyoku.github.io/2017/03/18/Deconvolution%E3%81%AE%E4%BB%A3%E3%82%8F%E3%82%8A%E3%81%ABPixel-Shuffler%E3%82%92%E4%BD%BF%E3%81%86/
class PixelShuffler(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = int(scale)
    
    def get_config(self):
        config = {'scale': self.scale}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, x, training=None):
        B,H,W,C = tf.keras.backend.int_shape(x)
        out_C   = int(C / self.scale / self.scale)
        x = tf.keras.layers.Reshape((H, W, self.scale, self.scale, out_C))(x)
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        x = tf.keras.layers.Reshape((H*self.scale, W*self.scale, out_C))(x)
        return x

    def compute_output_shape(self, input_shape):
        B, H, W, C = input_shape
        _H, _W     = H*self.scale, W*self.scale
        _C         = int(C / self.scale / self.scale)
        return (B, _H, _W, _C)