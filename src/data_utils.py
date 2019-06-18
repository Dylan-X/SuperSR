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