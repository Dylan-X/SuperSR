from .common import BaseSRModel
from tensorflow.python.keras import layers
from tensorflow.python import keras


class SRCNN_915(BaseSRModel):
    """
    Using 9-1-5 model.
    """

    def __init__(self, scale, model_name, channel=1):

        super(SRCNN_915, self).__init__(scale, model_name, channel)

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

    def create_model(self, load_weights=False, weights_path=None):
        '''
        XXX padding should be `valid` in training mode and `same` in testing mode.
        Here uses `same` in both cases for convenience of implementation.
        You can change it to `valid` if needed, don't forget to modify the labels' size.
        '''

        inp = super(SRCNN, self).create_model()

        x = layers.Convolution2D(self.n1, (self.f1, self.f1),
                                 activation='relu', padding='same', name='level1')(inp)
        x = layers.Convolution2D(self.n2, (self.f2, self.f2),
                                 activation='relu', padding='same', name='level2')(x)

        out = layers.Convolution2D(self.channel, (self.f3, self.f3),
                                   padding='same', name='output')(x)

        model = keras.Model(inp, out)

        if load_weights:
            weights_path = self.weights_path if weights_path is None else weights_path
            model.load_weights(self.weight_path)
            print("loaded model %s from %s" % (self.model_name, weights_path))

        self.model = model
        return self

    def lr_schedule(self, epoch, *args, **kwargs):
        # Following the original paper in which learning rate is set to 1e-5 for last layer.
        return 1e-5


class SRCNN_955(BaseSRModel):
    """
    Using 9-5-5 model.
    """

    def __init__(self, scale, model_name, channel=1):

        super(SRCNN_955, self).__init__(scale, model_name, channel)

        self.f1 = 9
        self.f2 = 5
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32
