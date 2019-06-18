from tensorflow.python.keras import layers, callbacks, optimizers
from tensorflow.python.keras.utils import plot_model
from tensorflow.python import keras
import tensorflow as tf
import os

from ..wn import AdamWithWeightnorm
from .utils import psnr_tf


class BaseSRModel(object):
    """Base model class of all models for SR.

        This is a base model class for super-resolution, contains creating and training of the model.
         If you want to create new models based on this, you need to complete the create_model() func. 

        Attributes:
            model_name: Name of this model.
            weights_path: Path to save this model, using "./weights/model_name.h5" by default.
            log_dir: Directory to save tensorboard log files.
            scale: Super-resolution ratio factor.
            inp_shape: Shape of input data in tuple, e.g. (None, None, 3).
            channel: Number of channels of both inputs and outputs.
            model: keras Model object.

        Methods:
            create_model(): XXX Generate the model, you need to complete this func.
            lr_schedule(): Define the learning rate with respect to epoch.
            fit(): train the model with training dataset and hyperparameters.
                - trdst: Tensorflow Dataset for training.
                - valdst: Tensorflow Dataset for validation.
                - nb_epochs: Int, number of epochs to train.
                - steps_per_epoch: Int, number of back propagations per epoch.
                - batch_size: Int, batch size.
                - use_wn: Whether to use Adam with Weight-Normalization when training. 
                          (Using Adam directly by default.)
            plot_model(): plot the model and save to ./
    """

    def __init__(self, scale, model_name, channel=3):
        self.inp_shape = (None, None, channel)
        self.channel = channel
        self.scale = scale
        self.model_name = "%s_X%d" % (model_name, scale)
        os.makedirs("./weights", exist_ok=True)
        self.weights_path = "./weights/%s_X%d.h5" % (model_name, scale)
        self.log_dir = "logs"
        self.model = None

    def create_model(self,
                     load_weights=False,
                     weights_path=None,
                     **kwargs):
        return layers.Input(self.inp_shape)

    def lr_schedule(self, epoch, max_epoch):
        if epoch < max_epoch // 2:
            return 1e-4
        elif epoch < max_epoch * 2 // 3:
            return 5e-4
        else:
            return 1e-5

    def fit(self,
            trdst,
            valdst,
            nb_epochs,
            steps_per_epoch,
            batch_size=100,
            use_wn=False):

        opt = AdamWithWeightnorm() if use_wn else optimizers.Adam()
        self.model.compile(optimizer=opt,
                           loss='mse', metrics=[psnr_tf])

        log_dir = os.path.join(self.log_dir, self.model_name)
        callback_list = [
            callbacks.ModelCheckpoint(
                self.weights_path,
                save_best_only=False,
                save_weights_only=True,
                verbose=1),
            callbacks.LearningRateScheduler(
                lambda e: self.lr_schedule(e, nb_epochs), verbose=0),
            callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1, write_graph=True)
        ]

        print('Training model : %s' % (self.model_name))

        self.model.fit(
            x=trdst.batch(batch_size),
            epochs=nb_epochs,
            callbacks=callback_list,
            validation_data=valdst.batch(batch_size),
            steps_per_epoch=steps_per_epoch,
            verbose=1)

        return self

    def plot_model(self, ):
        plot_model(
            self.model,
            to_file="./%s.png" % self.model_name,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB")
