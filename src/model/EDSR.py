import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from .common import BaseSRModel
from .utils import SubpixelLayer, MeanShift


class _ResBlock(keras.Model):
    def __init__(self, F, scale_f, *args, **kwargs):

        super(_ResBlock, self).__init__(*args, **kwargs)
        self.conv1 = layers.Conv2D(
            F, (3, 3), padding="same", activation='relu')
        self.conv2 = layers.Conv2D(F, (3, 3), padding="same")
        self.scale = layers.Lambda(lambda x: scale_f * x, name="scale")
        self.add = layers.Add(name="add")

    def call(self, inputs):
        x1 = self.scale(self.conv2(self.conv1(inputs)))
        return self.add([inputs, x1])


def EDSR_func(inp, scale, F, nb_res, res_scale_f):
    x = MeanShift(-1)(inp)
    x = layers.Conv2D(F, (3, 3), padding="same")(inp)
    conv1 = x
    for i in range(nb_res):
        x = _ResBlock(F, res_scale_f, name="res%d" % i)(x)
    x = layers.Conv2D(F, (3, 3), padding="same")(x)
    x = layers.Add()([conv1, x])
    if scale == 2 or scale == 3:
        x = SubpixelLayer(scale=scale, out_channel=F, kernel_size=3)(x)
    elif scale == 4:
        x = SubpixelLayer(scale=2, out_channel=F, kernel_size=3)(x)
        x = SubpixelLayer(scale=2, out_channel=F, kernel_size=3)(x)
    else:
        raise ValueError("Wrong value of scale factor.")
    out = layers.Conv2D(3, (3, 3), padding="same")(x)
    out = MeanShift(1)(out)
    return out


class EDSR(BaseSRModel):
    def __init__(self, scale, model_name, channel=3):
        super(EDSR, self).__init__(scale, model_name, channel)

        self.F = 256
        self.nb_resblock = 32
        self.res_scale_f = 0.1

    def create_model(self, load_weights=False, weights_path=None):
        inp = super(EDSR, self).create_model()
        out = EDSR_func(inp, scale=self.scale, F=self.F,
                        nb_res=self.nb_resblock, res_scale_f=self.res_scale_f)
        model = keras.Model(inp, out)

        if load_weights:
            weights_path = self.weights_path if weights_path is None else weights_path
            model.load_weights(self.weight_path)
            print("loaded model %s from %s" % (self.model_name, weights_path))

        self.model = model
        return self


class EDSR_baseline(EDSR):
    def __init__(self, scale, model_name, channel=3):
        super(EDSR_baseline, self).__init__(scale, model_name, channel)

        self.F = 64
        self.nb_resblock = 16
        self.res_scale_f = 1.0
