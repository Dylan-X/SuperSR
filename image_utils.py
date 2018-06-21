# image preprocessing
import os
import shutil
import sys
from ast import \
    literal_eval  # used to transform str to dic, because dic cannot be saved in h5file.
from PIL import Image
import h5py
import numpy as np
# from scipy.misc import imread, imresize, imsave
import matplotlib.pyplot as plt

MODE = {"BICUBIC": Image.BICUBIC,
        "NEAREST": Image.NEAREST, "BILINEAR": Image.BILINEAR}


############################################
# DOWNSAMPLE, BLOCK-EXTRACTION, BLOCK-MERGING.
############################################


def color_mode_transfer(image, mode):
    """Transform the color mode of image

        Mode of an image defines the type and depth of a pixel in the image. PIL supports the following standard modes:

            1 (1-bit pixels, black and white, stored with one pixel per byte)
            L (8-bit pixels, black and white)
            P (8-bit pixels, mapped to any other mode using a color palette)
            RGB (3x8-bit pixels, true color)
            RGBA (4x8-bit pixels, true color with transparency mask)
            CMYK (4x8-bit pixels, color separation)
            YCbCr (3x8-bit pixels, color video format)
            Note that this refers to the JPEG, and not the ITU-R BT.2020, standard
            LAB (3x8-bit pixels, the L*a*b color space)
            HSV (3x8-bit pixels, Hue, Saturation, Value color space)
            I (32-bit signed integer pixels)
            F (32-bit floating point pixels)

        Args:
            image: Image in numpy array.
            mode: String, one of the modes listed above.

        Returns:
            Image been transfered in numpy array.
    """
    img = Image.fromarray(image)
    img_new = img.new(mode)
    return np.array(img_new)


def modcrop(image, scale):
    """Crop the image to size which can be devided by scale.

        Modify the size of image to be devided be scale.

        Args:
            image: Numpy array or Image object.
            scale: Scale to be modified. For example : if image have shape of 10*10, scale is 3, then return a image in shape of 9*9, in which 9 is dividable by 3.

        Returns:
            Image in numpy array which has been modified.

        Raises:
            ValueError: An error occured when image is not in 2-D or 3-D.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = image.squeeze()
    size = image.shape[:2]
    size -= np.mod(size, scale)
    if len(image.shape) == 2:
        image2 = image[:size[0], :size[1]]
    elif len(image.shape) == 3:
        image2 = image[:size[0], :size[1], :]
    else:
        raise ValueError("Input image should be a 2-D or 3-D array!")

    return image2


def hr2lr(image, scale=2, shape=0, interp="BICUBIC", keepdim=False, return_both=False):
    """resample image to specified size

        Resample the given image to specified size, using interplotion, could be "BICUBIC", "NEAREST" and "BILINEAR". 

        Args:
            image: Numpy array or Image object.
            scale: Number of scaling factor, in INT.
            shape: Optional. Size to be resampled to. 
                   If 0, keep size of the lr_size. 
                   If 1, keep size of the hr_size. 
                   If tuple, size to be resize to. 
            interp: Mode of interplotion, could be string of "BICUBIC", "NEAREST" and "BILINEAR". We use "BICUBIC" by default.
            keepdim: Whether set the channel of gray image as 1.
            return_both: Whether return both of hr_img and lr_img.

        Returns:
            Image in numpy array.

        Raises:
            ValueError: An error occured when shape is not a tuple or 0 or 1.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    image = modcrop(image, scale)
    hr_size = list(image.shape[:2])
    lr_size = [int(x/scale) for x in hr_size]

    img = Image.fromarray(image)
    img1 = img.resize(lr_size, resample=MODE[interp])

    if isinstance(shape, tuple):
        img2 = img1.resize(shape, resample=MODE[interp])
    elif shape == 1:
        img2 = img1.resize(hr_size, resample=MODE[interp])
    elif not shape:
        img2 = img1
    else:
        raise ValueError("Shape should be tuple or 0 or 1!")

    img2 = np.array(img2)
    if keepdim and len(img2.shape) == 2:
        img2 = img2[:, :, np.newaxis]
    if return_both:
        return np.array(image), img2
    return img2


def modcrop_batch(batch, scale):
    """Do modcrop on a whole batch of images.

        Do modcrop on a batch of images. Image in batch could be in different size. But they share the same scale factor. See modcrop() for more details.

        Args:
            batch: A batch of images. Could be in a numpy array or a list of numpy arrays. If it's in numpy array, the first dimension should define the number of images.
            scale: Scale to be modified. For example : if image have shape of 10*10, scale is 3, then return a image in shape of 9*9, in which 9 is dividable by 3.

        Returns:
            List of images in numpy array.
    """
    return list(map(modcrop, list(batch), [scale for _ in range(len(batch))]))


def hr2lr_batch(batch, scale, shape=0, interp="BICUBIC", keepdim=False):
    """resample image to specified size

        Do hr2lr() on batch of images. Image in batch could be in different size, but they share the same scale factor, interplotion method and final shape. (If shape is 0, the return list of images still could be in different shape.)

        Args:
            batch: A batch of images. Could be in a numpy array or a list of numpy arrays or Image objects. If it's in numpy array, the first dimension should define the number of images.
            scale: Number of scaling factor, in INT.
            shape: Optional. Size to be resampled to.
                   If 0, keep size of the lr_size.
                   If 1, keep size of the hr_size.
                   If tuple, size to be resize to.
            interp: Mode of interplotion, could be string of "BICUBIC", "NEAREST" and "BILINEAR". We use "BICUBIC" by default.

        Returns:
            List of images in numpy array.
        Raises:
            ValueError: An error occured when shape is not a tuple or 0 or 1.
    """
    return list(map(lambda img: hr2lr(img, scale, shape, interp, keepdim), list(batch)))


def slice_normal(image, size, stride, to_array=False, merge=False, **kargs):
    """Slice a image to bunch of blocks.

        Using slicing to cut image into bunch of blocks in same shape. Blocks can overlap each other determined by stride. 

        Args:
            image: Numpy array or Image object.
            size: Int defines height and width of blocks.
            stride: Int, stride defines the length of step once a time. #FIXME This is a bad discription.
            to_array: Bool, set true if you want to return a numpy array instead of a list.
            merge: Bool, whether merge to whole image or not.

        Returns:
            List of blocks. Or a numpy array if to_array is true.
            Tuple of nx and ny if merge, which are used to merge blocks into whole image.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    blocks = []
    nx = ny = 0
    h, w = image.shape[0:2]
    if len(image.shape) == 2:
        image = image.reshape((h, w, 1))

    # slicing
    for x in range(0, h-size+1, stride):
        nx += 1
        ny = 0
        for y in range(0, w-size+1, stride):
            ny += 1
            block = image[x:x+size, y:y+size, :]
            blocks.append(block)
    # list or array
    if to_array:
        blocks = np.array(blocks)
    if merge:
        return blocks, (nx, ny)
    return blocks


def slice_random(image, size, stride, nb_blocks, seed=None, to_array=False, **kargs):
    """Slice a image into blocks randomly.

        Using slice_normal() to slice the image into blocks and choose a sublist of blocks randomly.

        Args:
            image: Numpy array or Image object.
            size: Tuple(Int) defines height and width of blocks.
            stride: Stride defines the length of step once a time. #FIXME This is a bad discription.
            nb_blocks: number of blocks you want to select.
            seed: seed of numpy.random
            to_array: Set true if you want to return a numpy array instead of a list.

        Returns:
            List of blocks. Or a numpy array if to_array is true.
    """
    if seed:
        np.random.seed(seed)

    blocks = slice_normal(image, size, stride, to_array=True)
    index = np.random.permutation(len(blocks))[:nb_blocks]
    blocks = blocks[index]
    if to_array:
        blocks = np.array(blocks)
    return blocks


def _is_redundance(subim, blocks, threshold):
    """Use MSE to decide if the subim is redundance to blocks."""
    mses = np.mean(np.square(np.array(blocks) - subim), axis=(1, 2))
    if np.sum(mses < threshold) == 0:
        return False
    else:
        return True


def slice_rr(image, size, stride, threshold, to_array=False, **kargs):
    """Slice a image into blocks with removing redundance.

        We slice the image into blocks and do some filtering. We define the redundance by this:
                If two blocks' MSE value is smaller than threshold, one of them is redundance.
            In this case, we discard blocks which are regarded as redundance.



        Args:
            image: Numpy array.
            size: Int defines height and width of blocks.
            stride: Stride defines the length of step once a time. #FIXME This is a bad discription
            threshold: Value of the threshold to discard blocks.    
            to_array: Set true if you want to return a numpy array instead of a list.

        Returns:
            List of blocks. Or a numpy array if to_array is true.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    blocks = []
    h, w = image.shape[:2]
    if len(image.shape) == 2:
        image = image.reshape((h, w, 1))

    for x in range(0, h - size + 1, stride):
        for y in range(0, w - size + 1, stride):
            subim = image[x: x + size, y: y + size, :]
            if not len(blocks) or not _is_redundance(subim, blocks, threshold):
                blocks.append(subim)

    if to_array:
        blocks = np.array(blocks)
    return blocks


"""The function below is deprecated. Because we find that the results of slice-first-mode and downsample-first-mode are exactly the same. So we use slice-first-mode by default to be more scalable."""

# def _slice_multiscale_rr(images, size, stride, threshold):
#     if not isinstance(images, list) and not isinstance(images, tuple):
#         raise ValueError("Input images should be a list or tuple!")
#     if len(images[0].shape) == 2:
#         images = [image.reshape((image.shape[0], image.shape[1], 1))
#                   for image in images]
#     _scales = [images[0].shape[0]//image.shape[0] for image in images]
#     assert _scales[0] == 1, "Why first value of scales is not 1 ?!"
#     blocks = [[] for _ in range(len(images))]
#     h, w = images[0].shape[:2]
#     for x in range(0, h - size + 1, stride):
#         for y in range(0, w - size + 1, stride):
#             subims = [images[i][x//_scales[i]: (x+size)//_scales[i], y//_scales[i]: (
#                 y+size)//_scales[i], :] for i in range(len(images))]
#             if not len(blocks[0]) or not _is_redundance(subims[0], blocks[0], threshold):
#                 for i, block in enumerate(blocks):
#                     block.append(subims[i])

#     blocks = [np.array(block) for block in blocks]
#     return blocks

# def slice_multiscale(images, size, stride, mode="NORMAL", nb_blocks=None, threshold=None):
#     """Slice a batch of images into blocks.

#         Slicing a batch of images which could be in different shape. If shape is not same, we will use the first image as standard and calculate the size and stride of other images' blocks by default. Or you can specify the size of them.

#         Args:
#             images: List of images. Length of images should be less than 10.
#             size: Int defines size of blocks of first image, which will be set as standard size. If the other image are in different shape with first image, we will scale the block size and stride as well.
#             stride: Length of step to slicing of the first image. See above for details.
#             mode: Slicing mode, could be one of the "NORMAL", "RANDOM", "RR"
#             nb_blocks: If slicing mode is "RANDOM", nb_blocks should be a integer. See slice_random() for details.
#             threshold: If slicing mode is "RR", threshold should be a integer. See slice_rr() for details.

#         Returns:
#             List contains all blocks of different image.

#         Raises:
#             ValueError: An error occured when input images if not a list.
#             ValueError: Name of kargs are incorrect.
#             ValueError: If nb_blocks or threshold is None when mode is "RANDOM" or "RR", this error raises.
#             ValueError: If mode is not in ["NORMAL", "RANDOM", "RR"].
#     """
#     if not isinstance(images, list):
#         raise ValueError("Input images should be a list!")
#     _scales = [images[0].shape[0]//image.shape[0] for image in images]
#     assert _scales[0] == 1, "Why first value of scales is not 1 ?!"
#     sizes = [size//scale for scale in _scales]
#     strides = [stride//scale for scale in _scales]

#     to_arrays = [True for _ in range(len(images))]

#     if mode == "NORMAL":
#         return list(map(slice_normal, images, sizes, strides, to_arrays))
#     elif mode == "RANDOM":
#         if nb_blocks:
#             nb_blockss = [nb_blocks for _ in range(len(images))]
#             seed = np.random.randint(10)
#             seeds = [seed for _ in range(len(images))]
#             return list(map(slice_random, images, sizes, strides, nb_blockss, seeds, to_arrays))
#         else:
#             raise ValueError("In \"RANDOM\", nb_blocks cannot be NoneType!")
#     elif mode == "RR":
#         if threshold:
#             return _slice_multiscale_rr(images, size, stride, threshold)
#         else:
#             raise ValueError("In \"RR\", threshold cannot be NoneType!")
#     else:
#         raise ValueError(
#             "Mode should be one of the \"NORMAL\", \"RANDOM\", \"RR\"!")


def _merge_gray_(images, size, stride):
    """merge subimages into whole image, only support gray-scale."""
    if len(images.shape) == 3:
        images = images[:, :, :, np.newaxis]
    sub_size = images.shape[1]
    nx, ny = size[0], size[1]
    # the img mosaic.
    img = np.zeros((sub_size*nx, sub_size*ny, 1))
    for idx, image in enumerate(images):
        i = idx % ny
        j = idx // ny
        img[j*sub_size:j*sub_size+sub_size, i *
            sub_size:i*sub_size+sub_size, :] = image
    img = img.squeeze()

    # the right and left matrix used for merging.
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

    # Merge to whole.
    out = transLeft.dot(img.dot(transRight))
    return out


def merge_to_whole(images, size, stride):
    """merge the subimages to whole image. 

        A merging algorithm to merge subimages into whole image.

        Args:
            images: List of subimages or a numpy array.
            size: nx and ny, see slice_normal() for details.
            stride: stride of slicing.

        Returns:
            Whole image in numpy array.
    """
    if isinstance(images, list):
        images = np.array(images)
    if len(images.shape) == 3:
        images = images[:, :, :, np.newaxis]
    channel = images.shape[-1]
    new_image = list(map(_merge_gray_, [images[:, :, :, i] for i in range(
        channel)], [size for _ in range(channel)], [stride for _ in range(channel)]))
    final_size = (new_image[0].shape[0], new_image[0].shape[1], channel)
    final_image = np.zeros(final_size)
    for i in range(channel):
        final_image[:, :, i] = new_image[i]

    return final_image


############################################
# SAMPLE OF IMAGE-PROCESSING FUNCTION.
############################################


def hrlr_sliceFirst(image, scale, slice_type, hr_size, hr_stride, lr_shape=0, interp="BICUBIC", nb_blocks=None, seed=None, threshold=None):
    """Generate hr and lr blocks of single image.

        Using slice first method to generate hr and lr blocks. Only RGB image is available. Given image in numpy array, we first slice it into blocks, then downsample them to lr-block. If you want write a image processing function, generate data and save to h5 file using 
            save_h5() func, please follow the rules below:

                - The first input should be a string defines the path to image.
                - Return data in dictionary. Key will be regarded as name of data, value should be a numpy array. 
                - Value of dictionary should be in the same length.(i.e. all data matches.) #FIXME Find a way to raiseError if they don't match.

        Args:
            image: File path of image, should be a valid path.
            scale: Int defines the downsample scale_factor. Or a list of scales.
            slice_type: One of the function of slice_normal, slice_random and slice_rr.
            hr_size: Tuple defines the height and width of hr blocks.
            hr_stride: Int defines the stride of slicing.
            lr_shape: Tuple defines the height and width of lr block or 0 for lr_size or 1 for hr_size.
            interp: String defines the interplotion method when downsampling.
            nb_blocks: Int defines the number of blocks only if slice_type is slice_random.
            seed: Numpy random seed only if slice_type is slice_random.
            threshold: Int defines the threshold to remove redundance only if slice_type is slice_rr.
            merge: Bool defines whether merge to whole image or not only if slice_type is slice_normal.

        Returns:
            Dictionary of hr and lr blocks.

        Raises:
            ValueError: An error occured when scale is not a integer or a list(tuple) of integers.
            ValueError: An Warning occured when hr_size is not dividable by scale.
    """
    hr_img = np.array(Image.open(image))
    hr_blocks = slice_type(hr_img, hr_size, hr_stride, to_array=True,
                           merge=False, nb_blocks=nb_blocks, seed=seed, threshold=threshold)
    if sum([hr_size % sc for sc in scale]):
        raise ValueError(
            "Hr_size should be dividable by scale of %s" % (str(scale)))
    data = {"hr": hr_blocks}
    if isinstance(scale, int):
        lr_blocks = hr2lr_batch(
            hr_blocks, scale, shape=lr_shape, interp=interp, keepdim=True)
        data["lr_%dX" % (scale)] = np.array(lr_blocks)
    elif isinstance(scale, list) or isinstance(scale, tuple):
        lr_blocks = [hr2lr_batch(
            hr_blocks, sc, shape=lr_shape, interp=interp, keepdim=True) for sc in scale]
        for i, sc in enumerate(scale):
            data["lr_%dX" % (sc)] = np.array(lr_blocks[i])
    else:
        raise ValueError(
            "Scale should be an integer or a list(tuple) of integers.")
    return data


"""The function below is deprecated. Because we find that the results of slice-first-mode and downsample-first-mode are exactly the same. So we use slice-first-mode by default to be more scalable."""
# def hrlr_downsampleFirst(image, scale, slice_type, hr_size, hr_stride, lr_shape=0, interp="BICUBIC", nb_blocks=None, seed=None, threshold=None):
#     """Generate hr and lr blocks.

#         Using slice first method to generate hr and lr blocks.

#         Args:
#             image: File path of image, should be a valid path.
#             scale: Int defines the downsample scale_factor. Or a list of scales.
#             slice_type: Slicing mode, could be one of the "NORMAL", "RANDOM", "RR".
#             hr_size: Tuple defines the height and width of hr blocks.
#             hr_stride: Int defines the stride of slicing.
#             lr_shape: Tuple defines the height and width of lr block or 0 for lr_size or 1 for hr_size.
#             interp: String defines the interplotion method when downsampling.
#             nb_blocks: Int defines the number of blocks only if slice_type is slice_random.
#             seed: Numpy random seed only if slice_type is slice_random.
#             threshold: Int defines the threshold to remove redundance only if slice_type is slice_rr.
#             merge: Bool defines whether merge to whole image or not only if slice_type is slice_normal.

#         Returns:
#             Dictionary of hr and lr blocks.

#         Raises:
#             ValueError: An error occured when scale is not a integer or a list(tuple) of integers.
#     """
#     hr_img = np.array(Image.open(image))
#     if isinstance(scale, int):
#         lr_img = hr2lr(hr_img, scale, lr_shape, interp, keepdim=True)
#         blocks = slice_multiscale([hr_img, lr_img], hr_size, hr_stride,
#                                   mode=slice_type, nb_blocks=nb_blocks, threshold=threshold)
#     elif isinstance(scale, list):
#         lr_img = [hr2lr(hr_img, sc, lr_shape, interp, keepdim=True)
#                   for sc in scale]
#         blocks = slice_multiscale([hr_img]+lr_img, hr_size, hr_stride,
#                                   mode=slice_type, nb_blocks=nb_blocks, threshold=threshold)
#     else:
#         raise ValueError(
#             "Scale should be an integer or a list(tuple) of integers.")

#     data = {"hr": blocks[0]}
#     for i, sc in enumerate(scale):
#         data["lr_%dX" % (sc)] = blocks[i+1]
#     return data


############################################
# GENERATE AND SAVE DATA, FLOW DATA FROM H5FILE
############################################

def data_generator(image_dir, preprocess, count=False, **kargs):
    """Generate data using python generator.

        Specify a preprocessing function, we do that func on all images in directory. One time a image.

        Args:
            image_dir: Directory of images under preprocessed. String.
            preprocess: A preprocessing function
                Input: 
                    First argument should be the path of image, usually read the image by PIL.Image.
                    Return a dictionay, kes are name of data and value of date should be in numpy array.
                    Other arguments can be alternative, and all feed by **kargs.
            count: Bool defines whether count the number of data or not.
            **kargs: Arguments that will be fed into preprocessing function.

        Yield:
            Same as the output of preprocessing function.
    """
    count_num = 0
    total_num = len(list(sorted(os.listdir(image_dir))))
    # print(total_num)
    for filename in sorted(os.listdir(image_dir)):
        count_num += 1
        data = preprocess(os.path.join(image_dir, filename), **kargs)
        # print(count_num)
        # verbose
        if not count_num % 2:
            sys.stdout.write('\r %d data have been generated, %d data remained.' % (
                count_num, total_num-count_num))
        if count:
            yield count_num, data
        else:
            yield data


def save_h5(image_dir, save_path, preprocess, **kargs):
    """Save data into h5 file.

        We will use preprocessing function to generate the data of images in image_dir, and save them to save_path. We save them one image a time, so this func support Big Data case. Data generated from image should be in a dictionary, whose keys are used to create name of dataset in h5File. The last dataset of H5File is called "num_blocks", defines the number of data in each dataset.

        Args:
            image_dir: Directory of images. String.
            save_path: Path to the h5 file. If doesn't exist, we will create one by default.
            preprocess: Preprocessing function,
                Input: First argument should be the name of the image. Usually read the image by PIL.Image modual.
                Output: Important!! It should return a dictionary, keys are the name of data and data should be numpy array.
            **kargs: Arguments that will be fed into preprocessing function.
    """
    with h5py.File(save_path, 'a') as hf:
        length_dst = {}
        shape_dst = {}
        for _, data in data_generator(image_dir, preprocess, count=True, **kargs):
            for key, value in data.items():
                if not (key in list(hf.keys())):
                    shape_dst[key] = tuple(list(value.shape)[1:])
                    hf.create_dataset(
                        key, (0,)+shape_dst[key], maxshape=(None,)+shape_dst[key])
                    length_dst[key] = 0
                # length_dst[key] += len(value)
                hf[key].resize(length_dst[key]+len(value), axis=0)
                hf[key][length_dst[key]: length_dst[key] + len(value)] = value
                length_dst[key] += len(value)

        hf.create_dataset("num_blocks", data=length_dst['hr'])
        print("Length of different datasets are : " + str(length_dst))


def _index_generator(N, batch_size, shuffle=True, seed=None):
    """Generate index permanantly."""
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


def image_flow_h5(h5_path, batch_size, big_batch_size=1000, shuffle=True, index=None):
    """Image flow from h5 file.

        Using python generator to generate data pairs from h5 file. In case of the data might has big size causing OOM error, we use this method to generate data one batch a time. By the way, we use big batch to accelerate the IO speed, because reading from h5 file frequently is too slow. We save a size of big_batch_size of data into memory and generate batch during every big batch. The index is used to specify which data you want to yield. If you are using keras.Model.fit_generator(), you should yield a batch of data pairs  (data, label) per time.

        Args:
            h5_path: String, the path of h5 file.
            batch_size: Int, the size of batch.
            big_batch_size: Int, size of big batch.
            shuffle: Bool, whether shuffle the data or not.
            index: Tuple of the index, defines the index of wanted batches. e.g. (0,2), means we want to yield the first and third dataset in h5 file. If None, yield all datasets. 

        Raises:
            AssertError: An error occured when length of different dataset in h5 file are different.
    """
    while True:
        with h5py.File(h5_path, 'r') as hf:
            keys = [key for key in hf.keys()][:-1]
            N = hf[keys[0]].shape[0]
            index_generator = _index_generator(
                big_batch_size, batch_size, shuffle)
            for i in range(N//big_batch_size):
                data = [
                    hf[key][i*big_batch_size:(i+1)*big_batch_size] for key in keys]
                for _ in range(big_batch_size//batch_size):
                    index_array, _, current_batch_size = next(index_generator)

                    batches = [
                        np.zeros((current_batch_size,)+tuple(list(dt.shape)[1:])) for dt in data]

                    for k, index_ in enumerate(index_array):
                        for i in range(len(data)):
                            batches[i][k] = data[i][index_]

                    if index:
                        yield tuple([batches[i]/255. for i in index])
                    else:
                        yield tuple(batches)


"""
The function below reads data from h5 directly to numpy array. It is deprecated because it doesn't support Big Data case. So we use image_flow_h5() instead to generate data one batch a time.
"""
# def image_from_h5(h5_path, index=None):
#     """Generate data from h5 file.

#         Read data in h5 file directly.

#         Args:
#             h5_path: Path to h5file.
#             index: Tuple of the index, defines the index of wanted batches. e.g. (0,2), means we want to yield the first and third dataset in h5 file. If None, yield all datasets.
#         Returns:
#             data in tuple.
#     """
#     with h5py.File(h5_path, 'r') as hf:
#         keys = [key for key in hf.keys()][:-1]
#         if index:
#             keys = [keys[i] for i in index]
#         data = [np.array(hf[key]) for key in keys]
#     return tuple(data)


def main():
    # generate and save data to h5 file.
    save_h5(image_dir="../Dataset/DIV2K_valid_HR", save_path="/media/mulns/F25ABE595ABE1A75/H5File/div2k_val_same_248X.h5",
            preprocess=hrlr_sliceFirst, scale=[2, 4, 8], slice_type=slice_normal,  hr_size=48, hr_stride=24, lr_shape=1, threshold=None, nb_blocks=None)

    # Generator of data from h5File.
    datagen = image_flow_h5("/media/mulns/F25ABE595ABE1A75/H5File/div2k_tr_same_248X.h5", batch_size=100,
                            big_batch_size=1000, shuffle=True, index=(0, 2))
    
    # Visualize the generator. One batch a time.
    batches = next(datagen)
    imgs = []
    for batch in batches:
        print(batch.shape)
        imgs.append(batch[10])
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(imgs[i].squeeze())
    plt.show()


if __name__ == '__main__':
    main()

# TODO(mulns): Change the name of `preprocess` to `your_func`.
# TODO(mulns): Add image_flow_dir and save_dir function.
