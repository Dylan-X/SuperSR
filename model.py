
from keras.models import Model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.core import Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from keras.utils.np_utils import to_categorical
from keras.utils.io_utils import HDF5Matrix
import keras.callbacks as callbacks
import keras.optimizers as optimizers
from keras import backend as K
from advanced import  SubPixelUpscaling,  TensorBoardBatch
from image_utils import Dataset, downsample, merge_to_whole
from utils import psnr_k, psnr_np
import numpy as np
import os
import warnings
import scipy.misc
import h5py
import tensorflow as tf

class BaseSRModel(object):
    def __init__(self, model_name, input_size, channel):
        """
        Input:
            model_name, str, name of this model
            input_size, tuple, size of input. 
            channel, int, num of channel of input. 
        """
        self.model_name = model_name
        self.weight_path=None
        self.input_size=input_size
        self.channel=channel
        self.model=self.create_model(load_weights=False)

    def create_model(self, load_weights=False) -> Model:

        init = Input(shape=self.input_size)
        return init

    def fit(self, 
            train_dst=Dataset('./test_image/'), 
            val_dst=Dataset('./test_image/'),
            big_batch_size=1000, 
            batch_size=16, 
            learning_rate=1e-4, 
            loss='mse', 
            shuffle=True,
            visual_graph=True, 
            visual_grads=True, 
            visual_weight_image=True, 
            multiprocess=False,
            nb_epochs=100, 
            save_history=True, 
            log_dir='./logs') -> Model:
        
        assert train_dst._is_saved(), 'Please save the data and label in train_dst first!'
        assert val_dst._is_saved(), 'Please save the data and label in val_dst first!'
        train_count = train_dst.get_num_data()
        val_count = val_dst.get_num_data()

        if self.model == None: self.create_model()

        adam = optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=adam, loss=loss, metrics=[psnr_k])

        callback_list = []
        callback_list.append(callbacks.ModelCheckpoint(self.weight_path, monitor='val_loss',
                                                        save_best_only=True, mode='min', 
                                                        save_weights_only=True, verbose=2))
        if save_history:
            callback_list.append(TensorBoardBatch(log_dir=log_dir, batch_size=batch_size, histogram_freq=1,
                                                    write_grads=visual_grads, write_graph=visual_graph, write_images=visual_weight_image))

        print('Training model : %s'%(self.model_name))

        self.model.fit_generator(train_dst.image_flow(big_batch_size=big_batch_size, batch_size=batch_size, shuffle=shuffle),
                                steps_per_epoch=train_count // batch_size + 1, epochs=nb_epochs, callbacks=callback_list, 
                                validation_data=val_dst.image_flow(big_batch_size=10*batch_size, batch_size=batch_size, shuffle=shuffle),
                                validation_steps=val_count // batch_size + 1, use_multiprocessing=multiprocess, workers=4)
        return self.model                     


    def gen_sr_img(self, test_dst=Dataset('./test_image/'), image_name='Spidy.jpg', save = False, verbose=0):
        """
        Generate the high-resolution picture with trained model. 
        Input:
            test_dst: Dataset, Instance of Dataset. 
            image_name : str, name of image.
            save : Bool, whether to save the sr-image to local or not.  
            verbose, int.
        Return:
            orig_img, bicubic_img, sr_img and psnr of the hr_img and sr_img. 
        """
        stride = test_dst.stride
        scale = test_dst.scale
        lr_size = test_dst.lr_size
        assert test_dst.slice_mode=='normal', 'Cannot be merged if blocks are not completed. '

        data, label, size_merge = test_dst._data_label_(image_name)
        output = self.model.predict(data, verbose=verbose)
        # merge all subimages. 
        hr_img = merge_to_whole(label, size_merge, stride = stride)
        lr_img = downsample(hr_img, scale=scale, lr_size=lr_size)
        sr_img = merge_to_whole(output, size_merge, stride = stride)
        if verbose == 1:
            print('PSNR is %f'%(psnr_np(sr_img, hr_img)))
        if save:
            scipy.misc.imsave(sr_img, './example/%s_SR.png'%(image_name))
        return hr_img, lr_img, sr_img, psnr_np(sr_img, hr_img)

    def evaluate(self, test_dst=Dataset('./test_image/'), verbose = 0) -> Model:
        """
        evaluate the psnr of whole images which have been merged. 
        Input:
            test_dst, Dataset. A instance of Dataset. 
            verbose, int. 
        Return:
            average psnr of images in test_path. 
        """
        PSNR = []
        test_path = test_dst.image_dir
        for _, _, files in os.walk(test_path):
            for image_name in files:
                # Read in image 
                _, _, _, psnr =  self.gen_sr_img(test_dst, image_name)
                PSNR.append(psnr)
        ave_psnr = np.sum(PSNR)/float(len(PSNR))
        print('average psnr of test images(whole) in %s is %f. \n'%(test_path, ave_psnr))
        return ave_psnr


class SRCNN(BaseSRModel):
    def __init__(self, model_type, input_size, channel):
        """
        Input:
            model_type, str, name of this SRCNN-net. 
            input_size, tuple, size of input layer. 
            channel, int, num of channels of input data. 
        """
        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/SRCNN Weights %s.h5" % (model_type)
        super(SRCNN, self).__init__("SRCNN"+model_type, input_size, channel)        
    
    def create_model(self, load_weights=False):

        init = super(SRCNN, self).create_model()

        x = Convolution2D(self.n1, (self.f1, self.f1), activation='relu', padding='same', name='level1')(init)
        x = Convolution2D(self.n2, (self.f2, self.f2), activation='relu', padding='same', name='level2')(x)

        out = Convolution2D(channels, (self.f3, self.f3), padding='same', name='output')(x)

        model = Model(init, out)

        if load_weights: 
            model.load_weights(self.weight_path)
            print("loaded model%s"%(self.model_name))
        self.model = model
        return model
    


        
class ResNetSR(BaseSRModel):
    def __init__(self, ):
        pass

class EDSR(BaseSRModel):
    def __init__(self, ):
        pass

