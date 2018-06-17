
from keras.models import Model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from keras.utils.np_utils import to_categorical
from keras.utils.io_utils import HDF5Matrix
import keras.callbacks as callbacks
import keras.optimizers as optimizers
from keras import backend as K
from advanced import TensorBoardBatch
# from image_utils import Dataset, downsample, merge_to_whole
from utils import PSNR, psnr, SubpixelConv2D
from PIL import Image
from image_utils_copy import *
import numpy as np
import os
import warnings
import scipy.misc
import h5py
import tensorflow as tf



class BaseSRModel(object):
    """Base model class of all models for SR.

        This is a base model class for super-resolution, contains creating, training and evaluation of the model. If you want to add some new models based on this, you need to complete the create_model() func. By the way, this model supports data preprocessing in image_utils.py

        Attributes:
            model_name: Name of this model.
            weight_path: Path to save this model, using "./weights/model_name.h5" by default.
            input_size: Size of input data in tuple, e.g. (48, 48, 3).
            channel: Number of channels of input.
            model: Model.

            create_model(): Generate the model, you need to complete this func.
            fit():  
    """

    def __init__(self, model_name, input_size, weight_dir=None):
        """        
            Args:
                model_name: Name of this model.
                input_size: Size of input data in tuple, e.g. (48, 48, 3).
                weight_dir: Directory to save the weights.
        """
        self.model_name = model_name
        if weight_dir:
            self.weight_path = os.path.join(
                weight_dir, "%s.h5" % (self.model_name))
        else:
            self.weight_path = "weights/%s.h5" % (self.model_name)
        self.input_size = input_size
        self.channel = self.input_size[-1]
        self.model = self.create_model(load_weights=False)

    def create_model(self, load_weights=False) -> Model:
        """Create model here, you need to complete this func."""
        init = Input(shape=self.input_size)
        return init

    def fit(self,
            tr_gen,
            val_gen,
            num_train,
            num_val,

            learning_rate=1e-4,
            loss='mse',
            nb_epochs=500,
            batch_size=100,

            visual_graph=True,
            visual_grads=True,
            visual_weight_image=True,

            multiprocess=False,
            save_history=True,
            log_dir='./logs') -> Model:
        """Trains the model on data generated batch-by-batch by a Python generator.

            The generator is run in parallel to the model, for efficiency.
            For instance, this allows you to do real-time data augmentation
            on images on CPU in parallel to training your model on GPU.

            Args:
                Arg: Discription #FIXME

            Raises:
                IOError: An error occured when Discription
        """

        if self.model == None:
            self.create_model()

        # adam = optimizers.Nadam()
        adam = optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer=adam, loss=loss, metrics=[PSNR])

        callback_list = []
        # callback_list.append(HistoryCheckpoint(history_fn))
        callback_list.append(callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNR',
                                                       save_best_only=True, mode='max', save_weights_only=True, verbose=2))
        if save_history:
            log_dir = os.path.join(log_dir, self.model_name)
            callback_list.append(TensorBoardBatch(log_dir=log_dir, batch_size=batch_size, histogram_freq=1,
                                                  write_grads=visual_grads, write_graph=visual_graph, write_images=visual_weight_image))  # FIXME why these parameters doen't work?

        print('Training model : %s' % (self.model_name))

        self.model.fit_generator(tr_gen,
                                 steps_per_epoch=num_train // batch_size + 1, epochs=nb_epochs, callbacks=callback_list,
                                 validation_data=val_gen,
                                 validation_steps=num_val // batch_size + 1, use_multiprocessing=multiprocess, workers=4)
        return self.model

    def gen_sr_img(self, image_path, hr_size, hr_stride, scale=4, lr_shape=0, save=False, save_name="test", verbose=0, **kargs):
        """Generate the high-resolution picture with trained model. 

            Input:
                image_name : str, name of image.
                save : Bool, whether to save the sr-image to local or not.  
                verbose, int. #FIXME

            Return:
                orig_img, bicubic_img, sr_img and psnr of the hr_img and sr_img. 
        """
        img = np.array(Image.open(image_path))
        hr_block, size_merge = slice_normal(img, hr_size, hr_stride, to_array=True, merge=True)
        lr_block = hr2lr_batch(hr_block, scale, lr_shape)
        sr_block = self.model.predict(lr_block, verbose=verbose)
        # merge all subimages.
        hr_img = merge_to_whole(hr_block, size_merge, stride=hr_stride)
        lr_img = hr2lr(hr_img, scale=scale, shape=1)
        sr_img = merge_to_whole(sr_block, size_merge, stride=hr_stride)
        if verbose == 1:
            print('PSNR is %f' % (psnr(sr_img, hr_img)))
        if save:
            scipy.misc.imsave('./example/%s_SR.jpg' % (save_name), sr_img)
        return (hr_img, lr_img, sr_img), psnr(sr_img, hr_img)

    def evaluate(self, test_path, verbose=0, **kargs) -> Model:
        """evaluate the psnr of whole images which have been merged. 
            Input:
                test_path. Path of test data.
                verbose, int. #FIXME
            Return:
                average psnr of images in test_path. 
        """
        PSNR = []
        for _, _, files in os.walk(test_path):
            for image_name in files:
                # Read in image
                _, psnr = self.gen_sr_img(os.path.join(test_path, image_name), verbose=verbose, **kargs)
                PSNR.append(psnr)
        ave_psnr = np.sum(PSNR)/float(len(PSNR))
        print('average psnr of test images(whole) in %s is %f. \n' %
              (test_path, ave_psnr))
        return ave_psnr, PSNR


class SRCNN(BaseSRModel):
    """
    pass
    """

    def __init__(self, model_type, input_size):
        """
        Input:
            model_type, str, name of this SRCNN-net. 
            input_size, tuple, size of input layer. e.g.(48, 48, 3)
        """
        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        super(SRCNN, self).__init__("SRCNN_"+model_type, input_size)

    def create_model(self, load_weights=False):

        init = super(SRCNN, self).create_model()

        x = Convolution2D(self.n1, (self.f1, self.f1),
                          activation='relu', padding='same', name='level1')(init)
        x = Convolution2D(self.n2, (self.f2, self.f2),
                          activation='relu', padding='same', name='level2')(x)

        out = Convolution2D(self.channel, (self.f3, self.f3),
                            padding='same', name='output')(x)

        model = Model(init, out)

        if load_weights:
            model.load_weights(self.weight_path)
            print("loaded model %s" % (self.model_name))

        self.model = model
        return model


class ResNetSR(BaseSRModel):
    """
    Under test. A little different from original paper. 
    """

    def __init__(self, model_type, input_size, scale):

        self.n = 64  # size of feature. also known as number of filters.
        self.mode = 2
        self.f = 3  # filter size
        # by diff scales comes to diff model structure in upsampling layers.
        self.scale = scale

        super(ResNetSR, self).__init__("ResNetSR_"+model_type, input_size)

    def create_model(self, load_weights=False, nb_residual=5):

        init = super(ResNetSR, self).create_model()

        x0 = Convolution2D(self.n, (self.f, self.f), activation='relu',
                           padding='same', name='sr_res_conv1')(init)

        x = self._residual_block(x0, 1)

        nb_residual = nb_residual-1
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2)

        x = Add()([x, x0])

        if self.scale != 1:
            x = self._upscale_block(x, 1)

        x = Convolution2D(self.channel, (self.f, self.f), activation="linear",
                          padding='same', name='sr_res_conv_final')(x)

        model = Model(init, x)
        if load_weights:
            model.load_weights(self.weight_path, by_name=True)
            print("loaded model %s" % (self.model_name))

        self.model = model
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        init = ip

        x = Convolution2D(self.n, (self.f, self.f), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(
            axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1")(x, training=mode)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(self.n, (self.f, self.f), activation='linear', padding='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(
            axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2")(x, training=mode)

        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip
        scale = self.scale

        ps_features = self.channel*(scale**2)
        x = Convolution2D(ps_features, (self.f, self.f), activation='relu',
                          padding='same', name='sr_upsample_conv%d' % (id))(init)
        x = SubpixelConv2D(input_shape=self.input_size, scale=scale)(x)
        return x


class EDSR(BaseSRModel):

    def __init__(self, model_type, input_size, scale):

        self.n = 256  # size of feature. also known as number of filters.
        self.f = 3  # shape of filter. kernel_size
        self.scale_res = 0.1  # used in each residual net
        # by diff scales comes to diff model structure in upsampling layers.
        self.scale = scale
        super(EDSR, self).__init__("EDSR_"+model_type, input_size)

    def create_model(self, load_weights=False, nb_residual=10):

        init = super(EDSR, self).create_model()

        x0 = Convolution2D(self.n, (self.f, self.f), activation='linear',
                           padding='same', name='sr_conv1')(init)

        x = self._residual_block(x0, 1, scale=self.scale_res)

        nb_residual = nb_residual - 1
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2, scale=self.scale_res)

        x = Convolution2D(self.n, (self.f, self.f),
                          activation='linear', padding='same', name='sr_conv2')(x)
        x = Add()([x, x0])

        x = self._upsample(x)

        out = Convolution2D(self.channel, (self.f, self.f),
                            activation="linear", padding='same', name='sr_conv_final')(x)

        model = Model(init, out)

        if load_weights:
            model.load_weights(self.weight_path, by_name=True)
            print("loaded model %s" % (self.model_name))

        self.model = model
        return model

    def _upsample(self, x):
        scale = self.scale
        assert scale in [2, 3, 4, 8], 'scale should be 2, 3 ,4 or 8!'
        x = Convolution2D(self.n, (self.f, self.f), activation='linear',
                          padding='same', name='sr_upsample_conv1')(x)
        if scale == 2:
            ps_features = self.channel*(scale**2)
            x = Convolution2D(ps_features, (self.f, self.f), activation='relu',
                              padding='same', name='sr_upsample_conv2')(x)
            x = SubpixelConv2D(input_shape=self.input_size, scale=scale)(x)
        elif scale == 3:
            ps_features = self.channel*(scale**2)
            x = Convolution2D(ps_features, (self.f, self.f), activation='linear',
                              padding='same', name='sr_upsample_conv2')(x)
            x = SubpixelConv2D(input_shape=self.input_size, scale=scale)(x)
        elif scale == 4:
            ps_features = self.channel*(2**2)
            for i in range(2):
                x = Convolution2D(ps_features, (self.f, self.f), activation='linear',
                                  padding='same', name='sr_upsample_conv%d' % (i+2))(x)
                x = SubpixelConv2D(
                    input_shape=self.input_size, scale=2, id=i+1)(x)
        elif scale == 8:
            # scale 4 by 2 times or scale 2 by 4 times? under estimate!
            ps_features = self.channel*(4**2)
            for i in range(2):
                x = Convolution2D(ps_features, (self.f, self.f), activation='linear',
                                  padding='same', name='sr_upsample_conv%d' % (i+2))(x)
                x = SubpixelConv2D(
                    input_shape=self.input_size, scale=4, id=i+1)(x)
        return x

    def _residual_block(self, ip, id, scale):

        init = ip

        x = Convolution2D(self.n, (self.f, self.f), activation='relu', padding='same',
                          name='sr_res_conv_' + str(id) + '_1')(ip)
        x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(self.n, (self.f, self.f), activation='relu', padding='same',
                          name='sr_res_conv_' + str(id) + '_2')(x)

        Lambda(lambda x: x * self.scale_res)(x)
        m = Add(name="res_merge_" + str(id))([x, init])

        return m
