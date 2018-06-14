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

MODE = {"BICUBIC":Image.BICUBIC, "NEAREST":Image.NEAREST, "BILINEAR":Image.BILINEAR}


############################################
### downsample, block_extraction, block_merging.
############################################

# TODO(mulns): feature-extraction, clustering, sparce-coding (2018.6.16)

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
            image: Numpy array.
            scale: Scale to be modified. For example : if image have shape of 10*10, scale is 3, then return a image in shape of 9*9, in which 9 is dividable by 3.
    
        Returns:
            Image in numpy array which has been modified.
        
        Raises:
            ValueError: An error occured when image is not in 2-D or 3-D.
    """
    image = image.squeeze()
    size = image.shape[:2]
    size -= np.mod(size, scale)
    if len(image.shape)==2:
        image2 = image[:size[0], :size[1]]
    elif len(image.shape)==3:
        image2 = image[:size[0], :size[1], :]
    else:
        raise ValueError("Input image should be a 2-D or 3-D array!")

    return image2

def hr2lr(image, scale=2, shape=0, interp="BICUBIC"):
    """resample image to specified size
    
        Resample the given image to specified size, using interplotion, could be "BICUBIC", "NEAREST" and "BILINEAR". 
    
        Args:
            image: Numpy array.
            scale: Number of scaling factor, in INT.
            shape: Optional. Size to be resampled to. 
                   If 0, keep size of the lr_size. 
                   If 1, keep size of the hr_size. 
                   If tuple, size to be resize to. 
            interp: Mode of interplotion, could be string of "BICUBIC", "NEAREST" and "BILINEAR". We use "BICUBIC" by default. 

        Returns:
            Image in numpy array.
    
        Raises:
            ValueError: An error occured when shape is not a tuple or 0 or 1.
    """
    image = modcrop(image, scale)
    hr_size = list(image.shape[:2])
    lr_size = [x/scale for x in hr_size]

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
    return list(map(modcrop, list(batch)))

def hr2lr_batch(batch, scale, shape=0, interp="BICUBIC"):
    """resample image to specified size

        Do hr2lr() on batch of images. Image in batch could be in different size, but they share the same scale factor, interplotion method and final shape. (If shape is 0, the return list of images still could be in different shape.)

        Args:
            batch: A batch of images. Could be in a numpy array or a list of numpy arrays. If it's in numpy array, the first dimension should define the number of images.
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
    return list(map(lambda img:hr2lr(img, scale, shape, interp), list(batch)))

# TODO(mulns): image slicing, merging, saving, image flow, and other high-level API. (2018.6.15)


# TODO(mulns): debuging main func implementation. (2018.6.17)