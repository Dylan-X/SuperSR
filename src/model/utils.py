from tensorflow.python import keras
from tensorflow.python.keras import layers, callbacks, optimizers
import tensorflow as tf

class SubpixelLayer(keras.Model):
    '''
    '''

    def __init__(self,
                 scale,
                 out_channel,
                 kernel_size,
                 activation=None,
                 *args,
                 **kwargs):
        super(SubpixelLayer, self).__init__(*args, **kwargs)

        self.conv = layers.Conv2D(
            out_channel * scale**2,
            kernel_size,
            padding='same',
            activation=activation)
        self.subpixel = layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))

    def call(self, inputs):
        return self.subpixel(self.conv(inputs))


class MeanShift(keras.layers.Layer):

    '''
    Meanshift layer for EDSR, add or substract the mean RGB value of
    DIV2K dataset. (Normalized to 0 -- 1)

    Attribute:
        sign: 1 or -1, positive for adding and negative for substracting.
        rgb_mean: Tensor, mean value of RGB channel.
    '''

    def __init__(self, sign=-1):
        super(MeanShift, self).__init__()
        self.sign = sign

    def build(self, input_shape):
        self.rgb_mean = tf.convert_to_tensor([0.4488, 0.4371, 0.4040],
                                             dtype=tf.float32)

    def call(self, input):
        if self.sign == -1:
            return tf.math.subtract(input, self.rgb_mean)
        else:
            return tf.math.add(input, self.rgb_mean)


def identity(y_true, y_pred):
    # return the value of output
    return y_pred


def psnr_tf(a, b):
    # return the psnr of normalized tensors
    return tf.image.psnr(a, b, 1.0)
