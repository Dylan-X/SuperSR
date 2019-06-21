from tqdm import tqdm
from PIL import Image
import numpy as np
import itertools
import operator
import random
import math
import os


def modcrop(image, scale):
    '''Crop image wrt super-resolution scale factor.

        Param:
            image: Tensor or Numpy array.
            scale: Int.
        Return:
            Cropped image.
    '''
    h, w = image.shape[:2]
    size = (h, w)
    size -= np.mod(size, scale)
    image = image[:size[0], :size[1], ...]
    return image


def center_crop(img, target_size):
    '''Crop center area in shape of `target_size`(tuple of integers) of `img`.
    '''
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, target_size))
    end = tuple(map(operator.add, start, target_size))
    slices = tuple(map(slice, start, end))
    return img[slices]


def rgb2ycbcr(image):
    '''Convert RGB image to YCbCr color space.

        Only available on normalized image (value range in 0 to 255)
    '''
    assert image.max() <= 1.0, "Max value of image should be 1.0"

    R, G, B = [image[..., i][..., np.newaxis] for i in range(3)]
    Y = 0.257 * R + 0.504 * G + 0.098 * B + 16 / 255.
    Cb = -0.148 * R - 0.291 * G + 0.439 * B + 128 / 255.
    Cr = 0.439 * R - 0.368 * G - 0.071 * B + 128 / 255.
    final_img = np.concatenate((Y, Cb, Cr), axis=-1)

    return final_img.astype("float32")
