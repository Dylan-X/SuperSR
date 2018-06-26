# -*- coding: UTF-8 -*-
#!/home/mulns/anaconda3/bin/python
#    or change to /usr/lib/python3
# Created by Mulns at 2018/6/24
# Contact : mulns@outlook.com
# Visit : https://mulns.github.io


import os
import sys
import h5py
import numpy as np
from image_utils import normalize_img


############################################
# DATA SAVING AND FLOWING BASED ON H5PY
############################################

"""
Considering the big data case, we using python generator to save data and generate data. 
During saving, we save data generated from your_func one image a time into h5 file. Each dataset in h5 file is a kind of data and the number of them should be the same. The last dataset is named "num_blocks" defines the number of each dataset.
During flowing, we flow the data from h5 file one batch a time. You can choose whether to generate permanantly or not, and which dataset you want to get. Considering reading from h5 file is slow, we set `big batch` to accelerate the speed of batch generating. 
"""


def data_generator(image_dir, your_func, count=False, **kargs):
    """Generate data from image using python generator, one image a time.

        Specify a preprocessing function, we do that func on all images in directory. One time a image.

        Args:
            image_dir: Directory of images under preprocessed. String.
            your_func: A preprocessing function.
                Input: 
                    First argument should be the path of image, usually read the image by PIL.Image.
                    Return a dictionay, keys are the name of data, and the value of date should be in numpy array.
                    Other arguments can be alternative, and all feed by **kargs.
            count: Bool defines whether count the number of data or not.
            **kargs: Arguments that will be fed into preprocessing function.

        Yield:
            Same as the output of your function.
    """
    count_num = 0
    total_num = len(list(sorted(os.listdir(image_dir))))
    # print(total_num)
    for filename in sorted(os.listdir(image_dir)):
        count_num += 1
        data = your_func(os.path.join(image_dir, filename), **kargs)
        # print(count_num)
        # verbose
        if not count_num % 2:
            sys.stdout.write('\r From data_generator in image_utils.py : %d data have been generated, %d data remained.' % (
                count_num, total_num-count_num))
        if count:
            yield count_num, data
        else:
            yield data


def save_h5(image_dir, save_path, your_func, **kargs):
    """Save data into h5 file based on data_generator().

        We will use your function to generate the data of images in image_dir, and save them to save_path. We save them one image per time, so this func support Big Data case. Data generated from image should be in a dictionary, whose keys are used to create name of dataset in h5File. The last dataset of H5File is called "num_blocks", defines the number of data in each dataset.

        Args:
            image_dir: Directory of images. String.
            save_path: Path to the h5 file. If doesn't exist, we will create one by default.
            your_func: Preprocessing function,
                Input: First argument should be the name of the image. Usually read the image by PIL.Image modual.
                Output: Important!! It should return a dictionary, keys are the name of data, and the data should be in numpy array.
            **kargs: Arguments that will be fed into your function.
    """
    with h5py.File(save_path, 'a') as hf:
        length_dst = {}
        shape_dst = {}
        for _, data in data_generator(image_dir, your_func, count=True, **kargs):
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
        keys = list(hf.keys())
        hf.create_dataset("num_blocks", data=length_dst[keys[0]])
        print("\n Length of different datasets are : " + str(length_dst))


def index_generator(N, batch_size, keep_batch_size=False, shuffle=True, seed=None, loop=False, epoch=3):
    """Generate the index of batch.

        Args:
            N: Int.
                Number of all data.
            batch_size: Int.
                Number of the size of batch.
            keep_batch_size: Bool.
                Whether to keep the size of batch. When we generate batch, it's possible to have some batch, such as the last batch, is not in size of batch_size. So if Not keep_batch_size, we will ignore that one batch.
            shuffle: Bool.
                Whether to shuffle the data. If shuffle, index will not be generated in order.
            seed: Int or None.
                If shuffle, seed defines the seed value of numpy.random class. If seed is None, then ignore the seed.
            loop: Bool.
                Whether generate batch in loop. 
            epoch: Int or None.
                If loop, epoch defines the number of loops. If None, this generator will generate batch permanantly.

        Yields:
            Index of batch and the number of batch.

        Raises:
            Warning: If loop is True and epoch is None, a warning will be occurred because it will be a dead loop.
    """
    if not loop:
        epoch = 1
    count = 0
    if shuffle:
        if seed:
            np.random.seed(seed)
        index = np.random.permutation(N)
    else:
        index = np.arange(N)
    while True:
        for i in range(0, N, batch_size):
            if i + batch_size > N:
                if keep_batch_size:
                    continue
                batch_index = index[i:]
            else:
                batch_index = index[i: i+batch_size]
            yield (batch_index, len(batch_index))
        if epoch:
            count += 1
            if count >= epoch:
                break


def image_flow_h5(h5_path, batch_size, keep_batch_size=False, big_batch_size=1000, shuffle=True, seed=None, loop=True, epoch=None, index=None, normalize=False):
    """Image flow from h5 file.

        Using python generator to generate data pairs from h5 file. In case of the data might has big size causing OOM error, we use this method to generate data one batch a time. By the way, we use big batch to accelerate the IO speed, because reading from h5 file frequently is too slow. We crash the data with big_batch_size into memory and generate batch during every big batch period. The index is used to specify which data you want to yield. If you are using keras.Model.fit_generator(), you should yield a batch of data pairs  (data, label) per time.

        Args:
            h5_path: String.
                The path of h5 file.
            batch_size: Int.
                The size of each batch.
            keep_batch_size: Bool.
                Whether to keep the size of batch. When we generate batch, it's possible to have some batch, such as the last batch, is not in size of batch_size. So if Not keep_batch_size, we will ignore that kind of batch.
            big_batch_size: Int or None.
                Size of big batch. If None, we will set it as the number of all data in h5 file, which may cause OOM if data is too large. (H5 file should have a key named "num_blocks" defines the number of data in each dataset.)
            shuffle: Bool.
                Whether shuffle the data or not. If shuffle, we will shuffle all data but not the data in big_batch.
            seed: Int or None.
                Numpy.Random.seed value. If None, no seed.
            loop: Bool.
                Whether generate batches in loop. 
            epoch: Int or None.
                If loop, epoch defines the number of loops. If None, this generator will generate batch permanantly.
            index: Tuple of the index, defines the index of wanted datasets. e.g. (0,2), means we want to yield the first and third dataset in h5 file. If None, yield all datasets. #FIXME change to [keys]
            normalize: Bool, whether normalize image data or not. We only support image in 8-bit numpy array currently. #FIXME

        Yields:
            Tuple of batches from diff datasets. Each element is a numpy array in shape of (current_batch_size, height, width [, channel]).

        Raises:
            Exception: An error occured when h5 file has no key named "num_blocks".
                    Raises:
            Warning: If loop is True and epoch is None, a warning will be occurred because it will be a dead loop.
    """
    with h5py.File(h5_path, 'r') as hf:
        keys = [key for key in hf.keys()]
        if 'num_blocks' in keys:
            keys = list(index) if index else[key for key in keys if key != "num_blocks"]
        else:
            raise Exception("The last key of h5File should be \"num_blocks\"!")
        N = int(hf['num_blocks'].value)

        big_batch_size = N if not big_batch_size else big_batch_size

        big_batch_gen = index_generator(
            N, big_batch_size, keep_batch_size=False, shuffle=shuffle, seed=seed, loop=loop, epoch=epoch)
        for big_batch_index, current_big_batch_size in big_batch_gen:
            big_batch = [hf[key][[i for i in big_batch_index]] for key in keys]
            batch_gen = index_generator(current_big_batch_size, batch_size,
                                        keep_batch_size=keep_batch_size, shuffle=shuffle, seed=seed, loop=False)
            for batch_index, _ in batch_gen:

                batches = [data[batch_index] for data in big_batch]

                batches = list(map(normalize_img, batches)
                               ) if normalize else batches
                
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

# TODO(mulns): Add image_flow_dir and save_dir function.
