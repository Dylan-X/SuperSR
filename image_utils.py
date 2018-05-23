# image preprocessing
import os
import shutil
import sys
from ast import \
    literal_eval  # used to transform str to dic, because dic cannot be saved in h5file.

import h5py
import numpy as np
from scipy.misc import imread, imresize, imsave


"""
Image downsampling. Support multi-images(same size) processing. 
"""


def is_gray(image):
    assert len(image.shape) in (
        2, 3), 'image shape error, should be 2 or 3 dimensions!'
    image = np.squeeze(image)
    if len(image.shape) == 3:
        return False
    return True


def is_patch(data):
    """
    pass
    """
    shape = data.shape
    if len(shape) == 4:
        return True
    elif len(shape) == 2:
        return False
    elif len(shape) == 3 and shape[-1] in (1, 3, 4):
        return False
    elif len(shape) == 3:
        return True
    else:
        print(
            'Data shape incorrect. should be ([N,], hight, width [,channel])')
        raise ValueError


def formulate(data):
    """
    pass
    """
    shape = data.shape
    if len(shape) == 4:
        return data
    elif len(shape) == 2:
        return data.reshape((1, shape[0], shape[1], 1))
    elif len(shape) == 3 and shape[-1] in (1, 3, 4):
        return data.reshape((1, shape[0], shape[1], shape[2]))
    elif len(shape) == 3:
        return data.reshape((shape[0], shape[1], shape[2], 1))
    else:
        print(
            'Data shape incorrect. should be ([N,], hight, width [,channel])')
        raise ValueError


def modcrop(image, scale):
    """
    Return the image which could be devided by scale.
    Edge of image would be discard.
    If image is grayscale, return 2-D numpy array.  
    If image is a patch of images with same size, return the patch of modified images. 
    Input:
        image : ndarray, 2 or 3 or 4-D numpy arr. 
        scale : int, scale to be divided. 
    Return:
        image : ndarray, modified image or images. 
        is_patch : whether the input is a patch of images.
    ***
    If input image or images is grayscale, channel dimension will be ignored. Return np arr with shape of (N, size, size)
    """
    image = np.squeeze(image)
    if not is_patch(image):
        size = image.shape[:2]
        size -= np.mod(size, scale)
        if not is_gray(image):
            image = image[:size[0], :size[1], :]
        else:
            image = image[:size[0], :size[1]]
    else:
        size = image.shape[1:3]
        size -= np.mod(size, scale)
        if len(image.shape) == 4:
            image = image[:, :size[0], :size[1], :]
        else:
            image = image[:, :size[0], :size[1]]

    return image, is_patch(image)


def downsample(image, scale, interp='bicubic', downsample_flag=None):
    """
    Down sample the image to 1/scale**2.
    Input: 
        image : numpy array with shape of ([N, ] size, size [, channel])  
        scale : int
            Scale to downsample.
        interp : str, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic' or 'cubic').
        lr_size : tuple or int or NoneType, the output size of lr_image. None if keep size after scaling. 
    Return:
        Image_hr which has been modcropped. 
        Image_lr with shape of 1/scale, which has been squeezed. (i.e. No dimension with length of 1)

    """
    image, is_patch_ = modcrop(image, scale)

    if is_patch_:
        assert len(image.shape) in (
            3, 4), 'modcrop output Wrong shape. If processing a patch of images, the shape of arr should be 3 or 4-D!'

        data = []
        label = []
        for _, img in enumerate(image):
            img_lr = imresize(img, 1/scale, interp=interp)
            if downsample_flag is not None:
                img_lr = imresize(img_lr, img.shape[:2], interp='bicubic')
            data.append(img_lr)
            label.append(img)
        return np.array(label), np.array(data)
    else:
        assert len(image.shape) in (
            2, 3), 'modcrop output Wrong shape. If processing a patch of images, the shape of arr should be 2 or 3-D!'

        image_lr = imresize(image, 1/scale, interp=interp)
        if downsample_flag is not None:
            image_lr = imresize(image_lr, image.shape[:2], interp='bicubic')
        return image, image_lr


"""
Image slicing in diff mode. Support single image or a pair of images processing. 
"""


def _slice(images, size=48, stride=24, scale=2):
    """
    Slice the image into blocks with stride of stride, which could be reconstructed.
    Input:
        images : list, with one or two ndarray, 2-D or 3-D numpy array, could be in different size. 
                If list has two image, should be in (hr-image, lr-image) order.  
        size : int, size of block of first image in images. 
        stride : int, stride of slicing of hr-image in images.
        scale : int, scale of lr-image(if exists)
    Return:
        N : int, number of blocks.
        datas : list, with one or two ndarray, 
                numpy array with shape of (N, size, size, channel), and channel will be 1 if image is in grayscale.
                If list has two patch of subimages, it will be in (hr-patch, lr-patch) order. 
        (nx, ny) : tuple of two integers, used to merge original image.
    """

    if len(images) == 1:
        image = images[0]
        blocks = []
        nx = ny = 0
        h, w = image.shape[0:2]
        if is_gray(image):
            image = image.reshape((h, w, 1))

        for x in range(0, h - size + 1, stride):
            nx += 1
            ny = 0
            for y in range(0, w - size + 1, stride):
                ny += 1
                subim = image[x: x + size, y: y + size, :]
                blocks.append(subim)
        N = len(blocks)
        data = np.array(blocks)
        return N, (data), (nx, ny)

    elif len(images) == 2:
        hr_patch = []
        lr_patch = []
        nx = 0
        ny = 0
        hr_image, lr_image = images
        h, w = hr_image.shape[0:2]
        h_, w_ = lr_image.shape[0:2]
        if is_gray(hr_image):
            hr_image = hr_image.reshape((h, w, 1))
            lr_image = lr_image.reshape((h_, w_, 1))

        for x in range(0, h - size + 1, stride):
            nx += 1
            ny = 0
            for y in range(0, w - size + 1, stride):
                ny += 1
                hr_subim = hr_image[x: x + size, y: y + size, :]
                lr_subim = lr_image[x//scale : (x+size)//scale, y//scale : (y+size)//scale, :]
                hr_patch.append(hr_subim)
                lr_patch.append(lr_subim)

        N = len(hr_patch)
        label = np.array(hr_patch)
        data = np.array(lr_patch)
        return N, (label, data), (nx, ny)
    else:
        print('Wrong size of images, length of which should be 1 or 2!')
        raise ValueError


def _is_redundance(subim, blocks, Threshold):
    """
    Use MSE to decide if the subim is redundance to blocks.
    With little MSE, comes to great similarity, which means
    there has been images similar to this one. 
    Input:
        subim : numpy array.
        blocks : list of numpy arr or a numpy arr.
        Threshold : int. Higher threshold means more likely to be redundance. 
    Return : 
        Bool.
    """
    mses = np.mean(np.square(np.array(blocks) - subim), axis=(1, 2))
    if np.sum(mses < Threshold) == 0:
        return False
    return True


def _slice_rm_redundance(images, size=48, stride=24, scale=2, threshold=50):
    """
    Slice the image into blocks with removing redundance, which cannot be reconstructed.
    Input:
        images : list, with one or two ndarray, 2-D or 3-D numpy array, could be in different size. 
                If list has two image, should be in (hr-image, lr-image) order. 
        size : int, size of block of hr-image
        stride : int, stride of slicing of hr-image
        threshold : int, threshold to decide the similarity of blocks, higher threshold value means
            more likely to be removed. 
        scale : scale of lr-image from hr-image
    Return:
        N : int, number of blocks.
        datas : list, with one or two ndarray, 
                numpy array with shape of (N, size, size, channel), and channel will be 1 if image is in grayscale.
                If list has two patch of subimages, it will be in (hr-patch, lr-patch) order.   
        NoneType. 
    """
    if len(images) == 1:
        image = images[0]
        blocks = []
        h, w = image.shape[0:2]
        if is_gray(image):
            image = image.reshape((h, w, 1))

        for x in range(0, h - size + 1, stride):
            for y in range(0, w - size + 1, stride):
                subim = image[x: x + size, y: y + size, :]
                if len(blocks) == 0 or not _is_redundance(subim, blocks, threshold):
                    blocks.append(subim)
        N = len(blocks)
        data = np.array(blocks)
        return N, (data), None
    elif len(images) == 2:
        hr_patch = lr_patch = []
        hr_image, lr_image = images
        h, w = hr_image.shape[0:2]
        h_, w_ = lr_image.shape[0:2]

        if is_gray(hr_image):
            hr_image = hr_image.reshape((h, w, 1))
            lr_image = lr_image.reshape((h_, w_, 1))

        for x in range(0, h - size + 1, stride):
            for y in range(0, w - size + 1, stride):
                hr_subim = hr_image[x: x + size, y: y + size, :]
                lr_subim = lr_image[x//scale : (x+size)//scale, y//scale : (y+size)//scale, :]
                if len(blocks) == 0 or not _is_redundance(hr_subim, hr_patch, threshold):
                    hr_patch.append(hr_subim)
                    lr_patch.append(lr_subim)
        N = len(hr_patch)
        label = np.array(hr_patch)
        data = np.array(lr_patch)
        return N, (label, data), None
    else:
        print('Wrong size of images, length of which should be 1 or 2!')
        raise ValueError


def _slice_random(images, size, stride, scale, num, seed=None):
    """
    Slicing the image randomly. 
    Input:
        images : list, with one or two ndarray, 2-D or 3-D numpy array, could be in different size. 
                If list has two image, should be in (hr-image, lr-image) order. 
        size : int, size of block
        stride : int, stride of slicing when slice normally
        num : int, number of blocks to generate
        seed : None or int, random seed
    Return:
        num : int, number of subimages
        datas : list, with one or two ndarray, 
                numpy array with shape of (N, size, size, channel), and channel will be 1 if image is in grayscale.
                If list has two patch of subimages, it will be in (hr-patch, lr-patch) order.
        NoneType. 
    """
    N, data, _ = _slice(images, size=size, stride=stride, scale=scale)
    if seed != None:
        np.random.seed(seed)
    index = np.random.permutation(N)[:num]
    if len(data) == 1:
        data = data[0][index]
        return num, (data), None
    elif len(data) == 2:
        label, data = data[0][index], data[1][index]
        return num, (label, data), None
    else:
        print('Wrong size of images, length of which should be 1 or 2!')
        raise ValueError

def im_slice(images, size, stride, scale, num=None, threshold=None, seed=None, mode='normal'):
    """
    With different mode, return different subimages. 
    See _slice, _slice_rm_redundance, _slice_random for details. 
    Inputs:
        image, list, with numpy array
        size, int or tuple
        stride, int
        num, int. If mode is random, num's value will decide the number of blocks to generate. 
        threshold, int. If mode is rm_redundance, threshold's value will decide the redundance threshold. 
                        Higher threshold value means more likely to be removed. 
        mode : str. It should be normal, random or rm_redundance. 
    """
    assert mode in ('random', 'rm_redundance',
                    'normal'), 'Wrong mode, mode should be random, rm_redundance or normal!'
    if isinstance(size, tuple):
        size = size[0]

    if mode == 'random':
        assert isinstance(num, int), 'param \'num\' should be integer!'
        return _slice_random(images, size, stride, scale, num, seed)
    elif mode == 'rm_redundance':
        assert isinstance(
            threshold, int), 'param \'threshold\' should be integer!'
        return _slice_rm_redundance(images, size, stride, scale, threshold)
    else:
        return _slice(images, size, stride, scale)


def _merge_gray_(images, size, stride):
    """
    merge the subimages to whole image in grayscale. 
    Input:
            images: numpy array of subimages 
            size : tuple, (nx, ny) which is from the func _slice. 
            stride : the stride of generating subimages. 
    Output:
            numpy array with the same shape of original image(after modcropping)
    """

    sub_size = images.shape[1]
    nx, ny = size[0], size[1]
    img = np.zeros((sub_size*nx, sub_size*ny, 1))
    for idx, image in enumerate(images):
        i = idx % ny
        j = idx // ny
        img[j*sub_size:j*sub_size+sub_size, i *
            sub_size:i*sub_size+sub_size, :] = image
    img = img.squeeze()

    transRight = np.zeros((sub_size*ny, sub_size + stride*(ny-1)))
    transLeft = np.zeros((sub_size*nx, sub_size + stride*(nx-1)))
    one = np.eye(sub_size, sub_size)
    for i in range(ny):
        transRight[sub_size*i:sub_size*(i+1), stride*i:stride*i+sub_size] = one
    transRight = transRight/np.sum(transRight, axis=0)

    for i in range(nx):
        transLeft[sub_size*i:sub_size*(i+1), stride*i:stride*i+sub_size] = one
    transLeft = transLeft/np.sum(transLeft, axis=0)
    transLeft = transLeft.T

    out = transLeft.dot(img.dot(transRight))

    return out


def merge_to_whole(images, size, stride):
    images = formulate(images)
    channel = images.shape[-1]
    assert channel in (1, 3, 4), 'Wrong channel of input images!'
    images_in_channel = []
    for i in range(channel):
        images_in_channel.append(_merge_gray_(
            formulate(images[:, :, :, i]), size=size, stride=stride))
    orig_image = np.array(images_in_channel)*255.
    return orig_image.transpose(1, 2, 0).squeeze()


"""
image_generator
"""


class Dataset(object):
    def __init__(self, image_dir, data_label_path=None):
        """
        if data and label have already saved, save_path 
        should be the path to h5 or dir of blocks.  
        """
        self.image_dir = os.path.abspath(image_dir)
        self.save_path = data_label_path
        if self.save_path != None:
            self._unpack()

    def config_preprocess(self,
                          num_img_max=100,
                          color_mode='F',

                          slice_mode='normal',
                          hr_size=48,
                          stride=24,
                          num_blocks=None,
                          threshold=None,
                          seed=None,

                          downsample_mode='bicubic',
                          scale=4,
                          lr_size=None):
        """
        Configure the preprocessing param. 
        """

        self.num_image = num_img_max
        self.image_color_mode = color_mode
        if self.image_color_mode == 'F':
            self.channel = 1
        elif self.image_color_mode == 'RGBA':
            self.channel = 4
        else:
            self.channel = 3

        self.slice_mode = slice_mode
        self.hr_size = hr_size
        self.stride = stride
        self.num = num_blocks
        self.threshold = threshold
        self.seed = seed

        self.downsample_mode = downsample_mode
        self.scale = scale
        assert hr_size % self.scale == 0, 'Hr size is not dividable by scale!'
        if lr_size == 'same':
            self.downsample_flag = 'same'
            self.lr_size = (self.hr_size, self.hr_size)
        elif lr_size == None:
            self.downsample_flag = None
            self.lr_size = (self.hr_size // self.scale, self.hr_size // self.scale)
        else:
            print('lr_size should be NoneType or "same!"')
            raise ValueError

        # these param will be changed when saving func or datagen func is called.
        self.save_path = None
        self.save_mode = None

        self.batch_size = None
        self.shuffle = None

        self._pack_up()

    def _pack_up(self):
        """
        package up the param of preprocessing to save together with data and label. 
        """

        self.image = {}
        self.image['num_image'] = self.num_image
        self.image['color_mode'] = self.image_color_mode
        self.image['channel'] = self.channel

        self.slice = {}
        self.slice['slice_mode'] = self.slice_mode
        self.slice['hr_size'] = self.hr_size  # int
        self.slice['stride'] = self.stride
        self.slice['num_blocks'] = self.num
        self.slice['threshold'] = self.threshold
        self.slice['seed'] = self.seed

        self.downsample = {}
        self.downsample['downsample_mode'] = self.downsample_mode
        self.downsample['scale'] = self.scale
        self.downsample['lr_size'] = self.lr_size  # tuple

    def _unpack(self):
        """
        Unpack configuration param from saved h5file or directory. 
        """
        if os.path.isdir(self.save_path):
            raise NotImplementedError
        elif os.path.isfile(self.save_path):
            with h5py.File(self.save_path, 'r') as hf:
                self.image = literal_eval(hf['image'].value)
                self.slice = literal_eval(hf['slice'].value)
                self.downsample = literal_eval(hf['downsample'].value)

            self.num_img_max = self.image['num_image']
            self.image_color_mode = self.image['color_mode']
            self.channel = self.image['channel']

            self.slice_mode = self.slice['slice_mode']
            self.hr_size = self.slice['hr_size']
            self.stride = self.slice['stride']
            self.num = self.slice['num_blocks']
            self.threshold = self.slice['threshold']
            self.seed = self.slice['seed']

            self.downsample_mode = self.downsample['downsample_mode']
            self.scale = self.downsample['scale']
            self.lr_size = self.downsample['lr_size']

    def _data_label_(self, image_name):
        """
        Generate data and label of single picture. 
            Read image from path.
            Slice image into blocks.
            Downsample blocks to lr blocks.
        Can be overwrited if use other ways to preprocess images. 
        Input:
            image_name to be processed.
        Return:
            Data and Label to be fed in CNN. 4-D numpy arr. 
            size_merge, tuple, used to merge the whole image if slicing normally. 
        """
        assert self.image_color_mode in ('F', 'RGB', 'YCbCr', 'RGBA'), "Wrong mode of color \
                                        which should be in ('F', 'RGB', 'YCbCr', 'RGBA')"
        # read image from image_path.
        image = imread(os.path.join(self.image_dir, image_name),
                       mode=self.image_color_mode).astype(np.float)

        # image downsampling.
        hr_img, lr_img = downsample(
            image, scale=self.scale, interp=self.downsample_mode, downsample_flag=self.downsample_flag)

        # image slicing.
        if self.lr_size[0] == self.hr_size:
            scale = 1
        else:
            scale = self.scale
        N, l_out, size_merge = im_slice((hr_img, lr_img), size=self.hr_size, stride=self.stride, scale=scale,
                                        num=self.num, threshold=self.threshold, seed=self.seed,
                                        mode=self.slice_mode)
        # formulate the data and label to 4-d numpy array and scale to (0, 1)
        label, data = l_out
        data = formulate(data) / 255.
        label = formulate(label) / 255.

        return data, label, N, size_merge

    def _save_H5(self, verbose=1):
        """
        Save the data and label of a dataset dirctory into h5 files. 
        Under enhancing !! Not scalable yet...
        """

        num_dataInH5File = 0
        count = 0
        dataDst_shape = (self.lr_size[0], self.lr_size[1], self.channel)
        labelDst_shape = (self.hr_size, self.hr_size, self.channel)

        with h5py.File(self.save_path, 'a') as hf:
            dataDst = hf.create_dataset('data', (0,)+dataDst_shape,
                                        maxshape=(None, )+dataDst_shape)
            labelDst = hf.create_dataset('label', (0,)+labelDst_shape,
                                         maxshape=(None, )+labelDst_shape)
            # read images in diretory and preprocess them.
            for filename in sorted(os.listdir(self.image_dir)):
                count += 1
                # generate subimages of data and label to save.
                data, label, N, _ = self._data_label_(filename)
                # print(data.shape, label.shape)
                # add subimages of this image into h5 file.
                dataDst.resize((num_dataInH5File + N, )+dataDst_shape)
                dataDst[num_dataInH5File: (
                    num_dataInH5File + N), :, :, :] = data
                labelDst.resize((num_dataInH5File + N, )+labelDst_shape)
                labelDst[num_dataInH5File: num_dataInH5File + N, :, :, :] = label
                num_dataInH5File += N

                if verbose == 1:
                    if count % 10 == 0:
                        sys.stdout.write('\r %d images have been written in h5 file, %d remained.' % (
                            count, self.num_image-count))
                if count >= self.num_image:
                    print('Finished! %d hr-images in %s have been saved to %s as %d subimages together with lr-mode' %
                          (self.num_image, self.image_dir, self.save_path, num_dataInH5File))
                    break

            hf.create_dataset('num_subimages', data=num_dataInH5File)
            # dict cannot be saved in h5file, use string instead.
            hf.create_dataset('image', data=str(self.image))
            hf.create_dataset('slice', data=str(self.slice))
            hf.create_dataset('downsample', data=str(self.downsample))

    def _save_dir(self, verbose=1):
        raise NotImplementedError

    def save_data_label(self, save_mode='h5', save_path=None, verbose=1):
        """
        Save data and label to h5 file or to a directory. 
        If saved, use this func to claim the link of saved file/dir to this instance. 
        Input:  
            save_mode : str, should be h5 or dir 
        """
        assert save_mode in ('h5', 'dir'), 'Save_mode should be h5 or dir. '

        self.save_mode = save_mode
        if save_path == None:
            if save_mode == 'h5':
                self.save_path = './h5_files/%s.h5' % (
                    self.image_dir.split('/')[-1])
            elif save_mode == 'dir':
                self.save_path = './Data_images/%s/' % (
                    self.image_dir.split('/')[-1])
        else:
            self.save_path = save_path

        if self._is_saved():
            print('Congratulation! %s already exists!' % (self.save_path))

            return None
        elif self.save_mode == 'h5':
            hf = h5py.File(self.save_path, 'a')
            hf.close()
            assert os.path.isfile(
                self.save_path), 'Save path should be a h5 file!'

            return self._save_H5(verbose=verbose)
        elif self.save_mode == 'dir':
            assert os.path.isdir(
                self.save_path), 'Save path should be a dirctory!'

            return self._save_dir(verbose=verbose)

    def _image_flow_from_h5(self, big_batch_size=1000):
        """
        A python generator, to generate patch of data and label. 
        Input:
            big_batch_size : None or int, 
                This is used to speed up generating data. Frequent IO operation from
                h5 file is slow, so we crush a big batch of data into memory and read 
                patch from numpy array.
                Value of big_batch_size shouldn't be too large in case of memory outrage or 
                too small in case of read from h5 file frequently. 

        """
        assert os.path.exists(
            self.save_path), 'Please save the data and label to %s' % (self.save_path)

        if self.shuffle:
            if big_batch_size != None:
                while True:
                    with h5py.File(self.save_path, 'r') as hf:
                        N = int(hf['num_subimages'].value)
                        index_generator = self._index_generator(big_batch_size)
                        for i in range(N//big_batch_size):
                            data = hf['data'][i *
                                              big_batch_size: (i+1)*big_batch_size]
                            label = hf['label'][i *
                                                big_batch_size: (i+1)*big_batch_size]
                            for j in range(big_batch_size//self.batch_size):
                                index_array, _, current_batch_size = next(
                                    index_generator)
                                batch_x = np.zeros(
                                    (current_batch_size,) + (self.lr_size[0], self.lr_size[1], self.channel))
                                batch_y = np.zeros(
                                    (current_batch_size,) + (self.hr_size, self.hr_size, self.channel))
                                for k, index in enumerate(index_array):
                                    batch_x[k] = data[index]
                                    batch_y[k] = label[index]
                                yield (batch_x, batch_y)
            else:
                while True:
                    with h5py.File(self.save_path, 'r') as hf:
                        N = int(hf['num_subimages'].value)
                        index_generator = self._index_generator(N)
                        index_array, _, current_batch_size = next(
                            index_generator)
                        batch_x = np.zeros(
                            (current_batch_size,) + (self.lr_size[0], self.lr_size[1], self.channel))
                        batch_y = np.zeros(
                            (current_batch_size,) + (self.hr_size, self.hr_size, self.channel))
                        for k, index in enumerate(index_array):
                            batch_x[k] = hf['data'][index]
                            batch_y[k] = hf['label'][index]
                        yield (batch_x, batch_y)
        else:
            while True:
                if big_batch_size != None:
                    with h5py.File(self.save_path, 'r') as hf:
                        for i in range(N//big_batch_size):
                            data = hf['data'][i *
                                              big_batch_size: (i+1)*big_batch_size]
                            label = hf['label'][i *
                                                big_batch_size: (i+1)*big_batch_size]
                            for j in range(big_batch_size//self.batch_size):
                                batch_x = data[j *
                                               self.batch_size:(j+1)*self.batch_size]
                                batch_y = label[j *
                                                self.batch_size:(j+1)*self.batch_size]
                                yield (batch_x, batch_y)
                else:
                    with h5py.File(self.save_path, 'r') as hf:
                        batch_x = hf['data'][j *
                                             self.batch_size:(j+1)*self.batch_size]
                        batch_y = hf['label'][j *
                                              self.batch_size:(j+1)*self.batch_size]
                        yield (batch_x, batch_y)

    def _image_flow_from_dir(self, big_batch_size=1000):
        raise NotImplementedError

    def image_flow(self, big_batch_size=1000, batch_size=16, shuffle=True):
        """
        Image Generator to generate images by batches. 
        Input:
            flow_mode: str, should be h5 or dir
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

        assert self._is_saved(), "Please save the data and label first! \nOr claim the link of saved file to this instance by call 'save_data_label' func!"

        if os.path.isfile(self.save_path):
            return self._image_flow_from_h5(big_batch_size=big_batch_size)
        elif os.path.isdir(self.save_path):
            return self._image_flow_from_dir(big_batch_size=big_batch_size)

    def _index_generator(self, N):
        batch_size = self.batch_size
        shuffle = self.shuffle
        seed = self.seed
        batch_index = 0
        total_batches_seen = 0

        while 1:
            if seed is not None:
                np.random.seed(seed + total_batches_seen)

            if batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            current_index = (batch_index * batch_size) % N

            if N >= current_index + batch_size:
                current_batch_size = batch_size
                batch_index += 1
            else:
                current_batch_size = N - current_index
                batch_index = 0
            total_batches_seen += 1

            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def get_num_data(self):
        assert self._is_saved(), 'Data hasn\'t been saved!'
        if os.path.isdir(self.save_path):
            raise NotImplementedError
        elif os.path.isfile(self.save_path):
            with h5py.File(self.save_path, 'r') as hf:
                num_data = int(hf['num_subimages'].value)
        return num_data

    def cancel_save(self):
        # delete the h5 file or saving dir.
        if self._is_saved():
            if os.path.isfile(self.save_path):
                os.remove(self.save_path)
            elif os.path.isdir(self.save_path):
                shutil.rmtree(self.save_path)

    def _is_saved(self):
        if self.save_path != None and os.path.exists(self.save_path):
            return True
        else:
            return False


if __name__ == "__main__":
    dst = Dataset('../datasets/Car_train/Car_train/')
    dst.config_preprocess(num_img_max=4, lr_size='same')
    # dst._data_label_('n02960352_2638.JPEG')
    dst.save_data_label(save_path='./test1.h5')
    with h5py.File('./test1.h5', 'r') as hf:
        data, label = np.array(hf['data']), np.array(hf['label'])
    import matplotlib.pyplot as plt
    from utils import psnr
    lr_sample = data[100].squeeze()
    hr_sample = label[100].squeeze()
    # lr_sample = imresize(lr_sample, hr_sample.shape[:2], interp='bicubic')/255.
    print(psnr(lr_sample, hr_sample))
    plt.subplot(121)
    plt.imshow(lr_sample, 'gray')
    plt.subplot(122)
    plt.imshow(hr_sample, 'gray')
    plt.show()