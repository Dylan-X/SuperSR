import tensorflow as tf
import numpy as np
import random
import os
from .data_utils import modcrop

TF_INTERP = {
    0: tf.image.ResizeMethod.BILINEAR,
    1: tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    2: tf.image.ResizeMethod.BICUBIC
}


def gaussian_kernel(size=5, sigma=2.):
    ''' Get a gaussian kernel.

        Param:
            size: int 
                odd value
            sigma: float 
                blur factor

        return: 
            (normalized) Gaussian kernel，in shape of (size, size)
    '''
    x_points = np.arange(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
    y_points = x_points[::-1]
    xs, ys = np.meshgrid(x_points, y_points)
    kernel = np.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    kernel = kernel / kernel.sum()
    return kernel.astype("float32")


def downsample_gaussian(Hr, scale, kernel_size, kernel_sigma):
    '''Downsample Hr image with gaussian kernel.

        Params:
            Hr: Tensor or Numpy array.
                Hr image to be downsampled.
            scale: Int.
                Scale factor of downsampling ratio.
            kernel_size: Int.
                Size of Gaussian kernel.
            kernel_sigma: Float.
                Blur factor of Gaussian kernel.

        Return:
            Downsampled lr and Modcropped hr, normalized into 0--1 in float32. 
    '''
    kernel = tf.convert_to_tensor(gaussian_kernel(kernel_size, kernel_sigma))
    kernel = tf.concat([kernel[:, :, tf.newaxis, tf.newaxis]] * 3, axis=-2)
    kernel = tf.cast(kernel, tf.float32)

    Hr = modcrop(tf.cast(Hr, tf.float32), scale)
    Hr_p = tf.pad(Hr, [[BKS // 2, BKS // 2]] * 2 + [[0, 0]], "SYMMETRIC")

    downsampled_Lr = tf.squeeze(
        tf.nn.depthwise_conv2d(Hr_p[tf.newaxis, ...],
                               kernel,
                               strides=[1, scale, scale, 1],
                               padding='VALID'))
    Lr = tf.clip_by_value(downsampled_Lr, 0., 255.)

    return Lr / 255., Hr / 255.


def downsample_interp(Hr, scale, interp=2):
    '''Downsample Hr image with specific interpolation method.

        XXX To be noted, we set `antialias` False by defaults. One can adjust it if needed.

        Params:
            Hr: Tensor or Numpy array.
                Hr image to be downsampled.
            scale: Int.
                Scale factor of downsampling ratio.
            interp: Int. 
                Interpolation method. One of 0: bilinear, 1: nearest-neighbor, 2: bicubic.

        Return:
            Downsampled lr and Modcropped hr, normalized into 0--1 in float32. 
    '''

    Hr = modcrop(tf.cast(Hr, tf.float32), scale)
    H, W = Hr.shape[:2]

    downsampled_Lr = tf.image.resize(Hr, [H // scale, W // scale],
                                     method=TF_INTERP[interp],
                                     antialias=False)

    Lr = tf.clip_by_value(downsampled_Lr, 0., 255.)

    return Lr / 255., Hr / 255.


def degrade_image(Hr,
                  scale,
                  method=-1,
                  restore_shape=False,
                  noise_level=None,
                  **kwargs):
    '''Degrade Hr image with specific method, such as downsampling and adding additive noise.

        Params:
            Hr: Tensor or Numpy array.
                Hr-image to be downsampled.
            scale: Int.
                Scale factor of downsampling ratio.
            method: Int. 
                Downsampling method. One of -1: gaussian, 0: bilinear, 1: nearest-neighbor, 2: bicubic.
                See `downsample_interp` and `downsample_gaussian` for details.
                XXX If not using gaussian, we set `antialias` False by defaults. One can adjust it if needed.
            restore_shape: Bool.
                Whether to restore the shape of lr-image as shape of hr-image. (i.e., SRCNN data preprocessing)
                XXX To be noted, we use `bicubic` method to upsample image by default.
            noise_level: Float or None.
                If `noise` is not None, additive gaussian noise will be added to downsampled lr-image.
                (After downsampling, before upsampling)
                XXX To be noted, noise_level denotes the standard deviation of noise wrt. RGB image in (0, 255).
            **kwargs: Dict.
                If `method` is -1, `kernel_size` and `kernel_sigma` should be given. 
                See `downsample_gaussian` for details.

        Return:
            Degraded lr and Modcropped hr, normalized into 0--1 in float32. 
    '''

    if method == -1:
        assert 'kernel_size' in kwargs.keys(
        ) and 'kernel_sigma' in kwargs.keys(
        ), "With Gaussian method, size and sigma of kernel should be given."

        lr, hr = downsample_gaussian(Hr, scale, kwargs['kernel_size'],
                                     kwargs['kernel_sigma'])

    else:
        assert method in [
            0, 1, 2
        ], "Only -1: gaussian, 0: bilinear, 1: nearest-neighbor, 2: bicubic are supported"

        lr, hr = downsample_interp(Hr, scale, method)

    if noise_level is not None:
        noise = tf.random.normal(lr.shape, stddev=1.0,
                                 dtype=tf.float32) * noise_level / 255.
        lr = tf.clip_by_value(lr + noise, 0., 1.)

    if restore_shape:
        lr = tf.image.resize(lr * 255., hr.shape[:2],
                             method=TF_INTERP[2]) / 255.
        lr = tf.clip_by_value(lr, 0., 1.)

    return lr, hr
