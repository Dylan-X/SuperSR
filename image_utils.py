# image preprocessing
from scipy.misc import imread, imresize, imsave
import h5py
import numpy as np
import os
import sys
import shutil
from ast import literal_eval # used to transform str to dic, because dic cannot be saved in h5file. 


"""
Image downsampling. Support multi-images(same size) processing. 
"""

def is_gray(image):
    assert len(image.shape) in (2, 3), 'image shape error, should be 2 or 3 dimensions!'
    image = np.squeeze(image)
    if len(image.shape) == 3 :
        return False
    return True

def is_patch(data):
    """
    pass
    """
    shape = data.shape
    if len(shape)==4:
        return True
    elif len(shape)==2:
        return False
    elif len(shape)==3 and shape[-1] in (1,3,4):
        return False
    elif len(shape)==3:
        return True
    else:
        print('Data shape incorrect. should be ([N,], hight, width [,channel])')
        raise ValueError

def formulate(data):
    """
    pass
    """
    shape = data.shape
    if len(shape)==4:
        return data
    elif len(shape)==2:
        return data.reshape((1, shape[0], shape[1], 1))
    elif len(shape)==3 and shape[-1] in (1,3,4):
        return data.reshape((1, shape[0], shape[1], shape[2]))
    elif len(shape)==3:
        return data.reshape((shape[0], shape[1], shape[2], 1))
    else:
        print('Data shape incorrect. should be ([N,], hight, width [,channel])')
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
        if len(image.shape)==4:
            image = image[:, :size[0], :size[1], :]
        else:
            image = image[:, :size[0], :size[1]]

    return image, is_patch(image)

def downsample(image, scale, interp='bicubic', lr_size=None):
    """
    Down sample the image to 1/scale**2.
    Input: 
        image : numpy array with shape of ([N, ] size, size [, channel])  
        scale : int
            Scale to downsample. 
        interp : str, optional
            Interpolation to use for re-sizing ('nearest', 'lanczos', 'bilinear',
            'bicubic' or 'cubic').
        lr_size : tuple or int, the output size of lr_image. None if keep size after scaling. 
    Return:
        Image with shape of 1/scale**2, which has been squeezed. (No dimension with length of 1)
         
    """
    image, is_patch_ = modcrop(image, scale)
    if lr_size != None :
        assert isinstance(lr_size, int) or isinstance(lr_size, tuple) or lr_size == 'same', "Wrong type of lr_size which should be int or tuple type or 'same'."
    if isinstance(lr_size, int):
        lr_size = (lr_size, lr_size)

    if is_patch_:
        assert len(image.shape) in (3, 4), 'modcrop output Wrong shape. If processing a patch of images, the shape of arr should be 3 or 4-D!'
        data = []
        for _, img in enumerate(image):
            img_lr = imresize(img, 1/scale, interp=interp)
            if lr_size is not None and lr_size != 'same':
                img_lr = imresize(img_lr, lr_size, interp='bicubic')
            elif lr_size == 'same':
                img_lr = imresize(img_lr, img.shape, interp='bicubic')
            data.append(img_lr)
        return np.array(data)
    else:
        assert len(image.shape) in (2, 3), 'modcrop output Wrong shape. If processing a patch of images, the shape of arr should be 2 or 3-D!'
        image_lr = imresize(image, 1/scale, interp=interp)
        if lr_size is not None and lr_size != 'same':
            image_lr = imresize(image_lr, lr_size, interp='bicubic')
        elif lr_size == 'same':
            image_lr = imresize(image_lr, image.shape, interp='bicubic')
        return image_lr
        





"""
Image slicing in diff mode. Only one image processing. 
"""
def _slice(image, size, stride):
    """
    Slice the image into blocks with stride of stride, which could be reconstructed.
    Input:
        image : ndarray, 2-D or 3-D numpy array
        size : int, size of block
        stride : int, stride of slicing. 
    Return:
        N : int, number of blocks.
        data : ndarray, numpy array with shape of (N, size, size, channel), and channel will be 1 if image is in grayscale.
        (nx, ny) : tuple of two integers, used to merge original image.
    """
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
            subim = image[x : x + size, y : y + size, :]        
            blocks.append(subim)
    N = len(blocks)
    data = np.array(blocks)
    return N, data, (nx, ny)

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
    mses = np.mean(np.square(np.array(blocks) - subim), axis = (1, 2))
    if np.sum(mses < Threshold) == 0:
        return False
    return True

def _slice_rm_redundance(image, size, stride, threshold):
    """
    Slice the image into blocks with removing redundance, which cannot be reconstructed.
    Input:
        image : ndarray, 2-D or 3-D numpy array 
        size : int, size of block
        stride : int, stride of slicing. 
        threshold : int, threshold to decide the similarity of blocks, higher threshold value means
            more likely to be removed. 
    Return:
        N : int, number of blocks.
        data : ndarray, numpy array with shape of (N, size, size, channel), and channel will be 1 if the image is in grayscale.
    """
    blocks = []
    h, w = image.shape[0:2]
    if is_gray(image):
        image = image.reshape((h, w, 1))

    for x in range(0, h - size + 1, stride):
        for y in range(0, w - size + 1, stride):
            subim = image[x : x + size, y : y + size, :]   
            if len(blocks) == 0 or not _is_redundance(subim, blocks, threshold):
                blocks.append(subim)
    N = len(blocks)
    data = np.array(blocks)
    return N, data, None

def _slice_random(image, size, stride, num, seed=None):
    """
    Slicing the image randomly. 
    Input:
        image : ndarray, 2-D or 3-D numpy arr
        size : int, size of block
        stride : int, stride of slicing when slice normally
        num : int, number of blocks to generate
        seed : None or int, random seed
    Return:
        data : ndarray, numpy array with shape of (num, size, size, channel), and channel will be 1 if the image is in grayscale.
    """
    N, data, _ = _slice(image, size=size, stride=stride)
    if seed != None:
        np.random.seed(seed)
    index = np.random.permutation(N)[:num]
    return num, data[index], None

def im_slice(image, size, stride, num=None, threshold=None, seed=None, mode='normal'):
    """
    With different mode, return different subimages. 
    See _slice, _slice_rm_redundance, _slice_random for details. 
    Inputs:
        image, ndarray
        size, int
        stride, int
        num, int. If mode is random, num's value will decide the number of blocks to generate. 
        threshold, int. If mode is rm_redundance, threshold's value will decide the redundance threshold. 
                        Higher threshold value means more likely to be removed. 
        mode : str. It should be normal, random or rm_redundance. 
    """
    assert mode in ('random', 'rm_redundance', 'normal'), 'Wrong mode, mode should be random, rm_redundance or normal!'
    if mode == 'random':
        assert isinstance(num, int), 'param \'num\' should be integer!'
        return _slice_random(image, size, stride, num, seed)
    elif mode == 'rm_redundance':
        assert isinstance(threshold, int), 'param \'threshold\' should be integer!'
        return _slice_rm_redundance(image, size, stride, threshold)
    else:
        return _slice(image, size, stride)

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
		img[j*sub_size:j*sub_size+sub_size, i*sub_size:i*sub_size+sub_size, :] = image
	img = img.squeeze()

	transRight = np.zeros((sub_size*ny, sub_size + stride*(ny-1)))
	transLeft = np.zeros((sub_size*nx, sub_size + stride*(nx-1)))
	one = np.eye(sub_size,sub_size)
	for i in range(ny):
		transRight[sub_size*i:sub_size*(i+1), stride*i:stride*i+sub_size] = one
	transRight = transRight/np.sum(transRight, axis = 0)

	for i in range(nx):
		transLeft[sub_size*i:sub_size*(i+1), stride*i:stride*i+sub_size] = one
	transLeft = transLeft/np.sum(transLeft, axis = 0)
	transLeft = transLeft.T

	out = transLeft.dot(img.dot(transRight))

	return out

def merge_to_whole(images, size, stride):
    images = formulate(images)
    channel = images.shape[-1]
    assert channel in (1,3,4), 'Wrong channel of input images!'
    images_in_channel = []
    for i in range(channel):
        images_in_channel.append(_merge_gray_(formulate(images[:,:,:,i]), size=size, stride=stride))
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
                stride=16,
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
        assert hr_size%self.scale == 0, 'Hr size is not dividable by scale!'
        if isinstance(lr_size, int):
            self.lr_size = (lr_size, lr_size)
        elif lr_size == 'same':
            self.lr_size = (self.hr_size, self.hr_size)
        elif lr_size == None:
            self.lr_size = (hr_size // self.scale, hr_size // self.scale)
        else:
            assert isinstance(lr_size, tuple), 'Wrong type of parameter --lr_size, should be int or tuple or None or "same"'
            self.lr_size = lr_size

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
        self.slice['hr_size'] = self.hr_size, #int
        self.slice['stride'] = self.stride
        self.slice['num_blocks'] = self.num
        self.slice['threshold'] = self.threshold
        self.slice['seed'] = self.seed

        self.downsample = {}
        self.downsample['downsample_mode'] = self.downsample_mode
        self.downsample['scale'] = self.scale
        self.downsample['lr_size'] = self.lr_size #tuple

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
        image = imread(os.path.join(self.image_dir, image_name), mode = self.image_color_mode).astype(np.float)

        # image slicing. 
        N, label, size_merge = im_slice(image, size=self.hr_size, stride=self.stride, \
                                        num=self.num, threshold=self.threshold, seed=self.seed, \
                                        mode=self.slice_mode)
        # image downsampling.
        data = downsample(label, scale=self.scale, interp=self.downsample_mode, lr_size=self.lr_size)

        # formulate the data and label to 4-d numpy array and scale to (0, 1)
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
            dataDst = hf.create_dataset('data', (0,)+dataDst_shape, \
                maxshape=(None, )+dataDst_shape)
            labelDst = hf.create_dataset('label', (0,)+labelDst_shape, \
                maxshape=(None, )+labelDst_shape)
            # read images in diretory and preprocess them. 
            for filename in sorted(os.listdir(self.image_dir)):
                count += 1
                # generate subimages of data and label to save. 
                data, label, N, _ = self._data_label_(filename) 
                
                # add subimages of this image into h5 file. 
                dataDst.resize((num_dataInH5File + N, )+dataDst_shape)
                dataDst[num_dataInH5File : (num_dataInH5File + N), :, :, :] = data
                labelDst.resize((num_dataInH5File + N, )+labelDst_shape)
                labelDst[num_dataInH5File : num_dataInH5File + N, :, :, :] = label
                num_dataInH5File += N
                
                if verbose == 1:
                    if count%10 == 0:
                        sys.stdout.write('\r %d images have been written in h5 file, %d remained.'%(count, self.num_image-count))
                if count >= self.num_image:
                    print('Finished! %d hr-images in %s have been saved to %s as %d subimages together with lr-mode'%\
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
                self.save_path = './h5_files/%s.h5'%(self.image_dir.split('/')[-1])
            elif save_mode == 'dir':
                self.save_path = './Data_images/%s/'%(self.image_dir.split('/')[-1])
        else: 
            self.save_path = save_path
        
        if self._is_saved():
            print('Congratulation! %s already exists!'%(self.save_path))

            return None
        elif self.save_mode == 'h5':
            hf = h5py.File(self.save_path, 'a')
            hf.close()
            assert os.path.isfile(self.save_path), 'Save path should be a h5 file!'

            return self._save_H5(verbose=verbose)
        elif self.save_mode == 'dir':
            assert os.path.isdir(self.save_path), 'Save path should be a dirctory!'

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
        assert os.path.exists(self.save_path), 'Please save the data and label to %s'%(self.save_path)

        if self.shuffle:
            if big_batch_size != None:
                while True:
                    with h5py.File(self.save_path, 'r') as hf:
                        N = int(hf['num_subimages'].value)
                        index_generator = self._index_generator(big_batch_size)
                        for i in range(N//big_batch_size):
                            data = hf['data'][i*big_batch_size : (i+1)*big_batch_size]
                            label = hf['label'][i*big_batch_size : (i+1)*big_batch_size]
                            for j in range(big_batch_size//self.batch_size):
                                index_array, _, current_batch_size = next(index_generator)
                                batch_x = np.zeros((current_batch_size,) + (self.lr_size[0], self.lr_size[1], self.channel))
                                batch_y = np.zeros((current_batch_size,) + (self.hr_size, self.hr_size, self.channel))
                                for k, index in enumerate(index_array):
                                    batch_x[k] = data[index]
                                    batch_y[k] = label[index]
                                yield (batch_x, batch_y)
            else:
                while True:
                    with h5py.File(self.save_path, 'r') as hf:
                        N = int(hf['num_subimages'].value)
                        index_generator = self._index_generator(N)
                        index_array, _, current_batch_size = next(index_generator)
                        batch_x = np.zeros((current_batch_size,) + (self.lr_size[0], self.lr_size[1], self.channel))
                        batch_y = np.zeros((current_batch_size,) + (self.hr_size, self.hr_size, self.channel))
                        for k, index in enumerate(index_array):
                            batch_x[k] = hf['data'][index]
                            batch_y[k] = hf['label'][index]
                        yield (batch_x, batch_y)
        else:
            while True:
                if big_batch_size != None:
                    with h5py.File(self.save_path, 'r') as hf:
                        for i in range(N//big_batch_size):
                            data = hf['data'][i*big_batch_size : (i+1)*big_batch_size]
                            label = hf['label'][i*big_batch_size : (i+1)*big_batch_size]
                            for j in range(big_batch_size//self.batch_size):
                                batch_x = data[j*self.batch_size:(j+1)*self.batch_size]
                                batch_y = label[j*self.batch_size:(j+1)*self.batch_size]
                                yield (batch_x, batch_y)
                else:
                     with h5py.File(self.save_path, 'r') as hf:
                            batch_x = hf['data'][j*self.batch_size:(j+1)*self.batch_size]
                            batch_y = hf['label'][j*self.batch_size:(j+1)*self.batch_size]
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