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
from PIL import Image
from .image_utils import normalize_img, reduce_mean_, hr2lr_batch, slice_normal, slice_random, slice_rr


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
            List of index of batch and the number of data in batch.

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
            yield (list(sorted(batch_index)), len(batch_index))
        if epoch:
            count += 1
            if count >= epoch:
                break


def _multidata_util_(index, batches):
    f_b = []
    for i, tag in enumerate(index):
        if isinstance(tag, (tuple, list)):
            f_b.append([])
            for j, subtag in enumerate(tag):
                if isinstance(subtag, (tuple, list)):
                    raise ValueError(
                        "Currently we only support two level embedding...")
                else:
                    f_b[i].append(batches[i+j])
        else:
            f_b.append(batches[i])
    return tuple(f_b)


def image_flow_h5(h5_path, batch_size, keep_batch_size=False, big_batch_size=1000, shuffle=True, seed=None, loop=True, epoch=None, index=None, normalize=False, reduce_mean=False):
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
            index: Tuple or List of the index, defines the batch you want to yield.
                ("lr", "hr", "sr") : return the lr, hr and sr batch in tuple.
                (["lr_1","lr_2"], ["hr1", "hr2"]) : return two targets contains multiple batches. Used for multi-inputs and multi-outputs model in keras.
            reduce_mean: Bool, whether subtract the mean value of RGB of batch.

        Yields:
            Tuple of batches from diff datasets. Each element is a numpy array in shape of (current_batch_size, height, width [, channel]). The order is decided by index.

        Raises:
            Exception: An error occured when h5 file has no key named "num_blocks".
                    Raises:
            Warning: If loop is True and epoch is None, a warning will be occurred because it will be a dead loop.
    """
    f_keys = []

    def _get_all_keys_(index):
        for i in index:
            if isinstance(i, (tuple, list)):
                _get_all_keys_(i)
            else:
                f_keys.append(i)
    if index:
        _get_all_keys_(index)

    with h5py.File(h5_path, 'r') as hf:
        keys = [key for key in hf.keys()]
        if 'num_blocks' in keys:
            keys = f_keys if index else [
                key for key in keys if key != 'num_blocks']
        else:
            raise Exception("The last key of h5File should be \"num_blocks\"!")
        N = int(hf['num_blocks'].value)

        big_batch_size = N if not big_batch_size else big_batch_size

        big_batch_gen = index_generator(
            N, big_batch_size, keep_batch_size=False, shuffle=False, seed=seed, loop=loop, epoch=epoch)
        for big_batch_index, current_big_batch_size in big_batch_gen:
            big_batch = [hf[key][[i for i in big_batch_index]] for key in keys]
            batch_gen = index_generator(current_big_batch_size, batch_size,
                                        keep_batch_size=keep_batch_size, shuffle=shuffle, seed=seed, loop=False)
            for batch_index, _ in batch_gen:

                batches = [data[batch_index] for data in big_batch]

                batches = list(map(normalize_img, batches)
                               ) if normalize else batches
                batches = list(map(reduce_mean_, batches)
                               ) if reduce_mean else batches

                yield _multidata_util_(index, batches)


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


class _FlowData_(object):
    """This is a base class of data_flow.

         You can inherent your own class with specific processing function. before_save function is the function called before saving blocks into h5 file or directory, before_flow function is the function called before flowing data into network.

        Attributes:
            path: The path to h5 file or directory, as save path and flow path.
    """

    def __init__(self, path):
        if os.path.isdir(path):
            self._from_dir_ = True
        elif path.split(".")[-1] == "h5":
            self._from_h5_ = True
        else:
            raise NotImplementedError(
                "Currently only support flow from directory and h5 file.")
        self.flow_path = path

    def flow_from_h5(self, 
                     batch_size, 
                     keep_batch_size=False, 
                     big_batch_size=1000, 
                     shuffle=True, 
                     seed=None, 
                     loop=True, 
                     epoch=None, 
                     index=None):
        """Image flow from h5 file.

            Using python generator to generate data pairs from h5 file. In case of the data might has big size causing OOM error, we use this method to generate data one batch a time. By the way, we use big batch to accelerate the IO speed, because reading from h5 file frequently is too slow. We crash the data with big_batch_size into memory and generate batch during every big batch period. The index is used to specify which data you want to yield. If you are using keras.Model.fit_generator(), you should yield a batch of data pairs (data, label) per time.

            Args:
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
                index: Tuple or List of the index, defines the batch you want to yield.
<<<<<<< HEAD
                    ("lr", "hr", "sr") : return the lr, hr and sr batch in tuple.
                    (["lr_1","lr_2"], ["hr1", "hr2"]) : return two targets contains multiple batches. Used for multi-inputs and multi-outputs model in keras.
                **kargs:
                    See before_flow for details...
=======
>>>>>>> 29675d28675f3329394a176d46dca883c25d0bfd

            Yields:
                Tuple of batches from diff datasets. Each element is a numpy array in shape of (current_batch_size, height, width [, channel]). The order is decided by index.

            Raises:
                Exception: An error occured when h5 file has no key named "num_blocks".
        """

        with h5py.File(self.flow_path, 'r') as hf:
            keys = [key for key in hf.keys()]
            if 'num_blocks' in keys:
                keys = index if index else [
                    key for key in keys if key != 'num_blocks']
            else:
                raise Exception(
                    "The last key of h5File should be \"num_blocks\"!")
            N = int(hf['num_blocks'].value)

            big_batch_size = N if not big_batch_size else big_batch_size

            big_batch_gen = index_generator(
                N, big_batch_size, keep_batch_size=False, shuffle=False, seed=seed, loop=loop, epoch=epoch)
            for big_batch_index, current_big_batch_size in big_batch_gen:
                big_batch = [hf[key][[i for i in big_batch_index]]
                             for key in keys]
                batch_gen = index_generator(current_big_batch_size, batch_size,
                                            keep_batch_size=keep_batch_size, shuffle=shuffle, seed=seed, loop=False)
                for batch_index, _ in batch_gen:

                    batches = [data[batch_index] for data in big_batch]
                    fb = self.before_flow(batches)
                    assert isinstance(
                        fb, tuple), "before_flow should return a tuple."
                    yield fb

    def before_flow(self, batches):
        """Function called before flowing data.

            This function recieve a list of diff batches to be processed. Each batch is in numpy array or a list of numpy array. You should complete this function and return a tuple instance.

            Args:
                batches: List, different batches in numpy array or list of numpy array.

            Returns:
                Tuple of more than 2 elements.
        """
        return batches

    def _data_generator_(self, image_dir):
        """Generate data from image using python generator, one image a time.

            Specify a preprocessing function, we do that func on all images in directory. One time a image.

            Args:
                image_dir: Directory of images under preprocessed. String.
                your_func: A preprocessing function.

            Yield:
                Same as the output of your function.
        """
        count_num = 0
        total_num = len(list(sorted(os.listdir(image_dir))))
        # print(total_num)
        for filename in sorted(os.listdir(image_dir)):
            count_num += 1
            data = self.before_save(os.path.join(image_dir, filename))
            # print(count_num)
            # verbose
            if not count_num % 2:
                sys.stdout.write('\r From data_generator in DataFlow : %d data have been generated, %d data remained.' % (
                    count_num, total_num-count_num))

            yield count_num, data

    def save_to_h5(self, image_dir):
        """Save data into h5 file based on data_generator().

            We will use your function to generate the data of images in image_dir, and save them to save_path. We save them one image per time, so this func support Big Data case. Data generated from image should be in a dictionary, whose keys are used to create name of dataset in h5File. The last dataset of H5File is called "num_blocks", defines the number of data in each dataset.

            Args:
                image_dir: Directory of images. String.
                save_path: Path to the h5 file. If doesn't exist, we will create one by default.
        """
        with h5py.File(self.flow_path, 'a') as hf:
            length_dst = {}
            shape_dst = {}
            for _, data in self._data_generator_(image_dir):
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
            if not "num_blocks" in keys:
                hf.create_dataset("num_blocks", data=length_dst[keys[0]])
            print("\n Length of different datasets are : " + str(length_dst))

    def before_save(self, image_path):
        """Function called before saving data.

            This function recieve a string of image_path, returns a dictionary whose key is the name of dataset or directory. You should implement this function, to process a single image.

            Args:
                image_path: String.
                    Path to image.

            Returns:
                Dictionary: key is the name of dataset and directory. value is the data to be saved, should be in numpy array.

            Raises:
                IOError: An error occured when Discription
        """
        image = np.array(Image.open(image_path))
        return image

    def get_num_blocks(self):
        if self._from_h5_:
            with h5py.File(self.flow_path, 'r') as hf:
                return int(hf["num_blocks"].value)
        else:
            raise NotImplementedError("Currently only support h5 file..")


class SRFlowData(_FlowData_):
    """Super-Resolution base data flowing class.

        This class is used for common flowing data in Super-Resolution problem. We save blocks sliced from origin-image and save them to h5 file or directory. And flow data-pairs in tuple : (lr-blocks, hr-blocks).

        Attributes:
            h5path: String.
                Path to h5 files.
            bf_scale: Int.
                Downsample scaling_factor.
            bf_lr_shape: Int.
                0 to downsample.
                1 to downsample and upsample to size of origin-image.
            bs_slice_mode: String.
                "random" to slice random from origin-image.
                "normal" to slice without merging param.
                "rr" to slice with remove redundance.
            bs_block_size: Int.
                Value of width and height of blocks.
            bs_stride: Int.
                Stride when slicing blocks.
            bs_nb_blocks: Int.
                If slice_mode is "random", nb_blocks defines the number of blocks sliced from origin-image.
            bs_threshold: Int.
                If slice_mode is "rr", threshold defines the threshold of mse value to discard redundant blocks.

    """

    def __init__(self,
                 h5path,
                 bf_scale=None,
                 bf_lr_shape=None,
                 bs_slice_mode="normal",
                 bs_block_size=48,
                 bs_stride=24,
                 bs_nb_blocks=None,
                 bs_threshold=None):
        super(SRFlowData, self).__init__(h5path)
        self.scale = bf_scale
        self.lr_shape = bf_lr_shape
        self.slice_mode = bs_slice_mode
        self.block_size = bs_block_size
        self.stride = bs_stride
        self.nb_blocks = bs_nb_blocks
        self.threshold = bs_threshold

    def before_flow(self, batches):

        scale = self.scale
        lr_shape = self.lr_shape

        if scale and lr_shape is not None:

<<<<<<< HEAD
            if len(batches)==2:
                print("Please make sure u are reading lr and hr batch from h5 file.")
=======
            if len(batches) == 2:
                # print("Please make sure u are reading lr and hr batch from h5 file.")
>>>>>>> 29675d28675f3329394a176d46dca883c25d0bfd
                return tuple(batches)
            elif len(batches) == 1:
                # print("Please make sure u are reading hr batch from h5 file.")
                hr_batch = batches[0]
                lr_batch = np.array(hr2lr_batch(
                    hr_batch, scale=scale, shape=lr_shape, keepdim=True))
            else:
                raise ValueError(
                    "Wrong length of batches, if you want to process hr_batch differently, rewrite this funciton.")
            return tuple([lr_batch/255., hr_batch/255.])

        else:
            raise ValueError(
                "When calling flow_from_h5, scale and lr_shape should be specified.")

    def before_save(self, image_path):

        slice_mode = self.slice_mode
        block_size = self.block_size
        stride = self.stride
        nb_blocks = self.nb_blocks
        threshold = self.threshold

        if slice_mode:
            img = super(SRFlowData, self).before_save(image_path)
            if slice_mode == "random":
                if nb_blocks:
                    blocks = slice_random(
                        img, size=block_size, stride=stride, nb_blocks=nb_blocks, to_array=True)
                else:
                    raise ValueError(
                        "If slice_mode is random, nb_blocks should be specified in Integer.")
            elif slice_mode == "normal":
                blocks = slice_normal(
                    img, size=block_size, stride=stride, to_array=True, merge=False)
            elif slice_mode == "rr":
                if threshold:
                    blocks = slice_rr(
                        img, size=block_size, stride=stride, threshold=threshold, to_array=True)
                else:
                    raise ValueError(
                        "If slice_mode is rr which means remove redundance, threshold should be specified in Integer.")
            else:
                raise ValueError(
                    "Slice mode should be in 'random', 'normal', 'rr'.")

            return {'hr': blocks}

        else:
            raise ValueError(
                "When calling save_to_h5, slice_mode should be specified, which can be 'random', 'normal', 'rr'. So as stride and block_size.")


class MultiDataSRFLowData(SRFlowData):

    def __init__(self,
                 h5path,
                 bf_lr_shape=None,
                 bf_index=(2, [1, 1]),
                 bs_slice_mode="normal",
                 bs_block_size=48,
                 bs_stride=24,
                 bs_nb_blocks=None,
                 bs_threshold=None):
        super(MultiDataSRFLowData, self).__init__(h5path=h5path,
                                                  bf_scale=None,
                                                  bf_lr_shape=bf_lr_shape,
                                                  bs_slice_mode=bs_slice_mode,
                                                  bs_block_size=bs_block_size,
                                                  bs_stride=bs_stride,
                                                  bs_nb_blocks=bs_nb_blocks,
                                                  bs_threshold=bs_threshold)
        self.index = bf_index

    def before_flow(self, batches):

        lr_shape = self.lr_shape
        index = self.index

        if lr_shape is not None:

            hr_batch = batches[0]
            f_keys = []

            def _get_all_keys_(index):
                for i in index:
                    if isinstance(i, (tuple, list)):
                        _get_all_keys_(i)
                    else:
                        assert isinstance(
                            i, int), "value in index should be Integer."
                        f_keys.append(i)
            if index:
                _get_all_keys_(index)

            _batches_ = []
            for scale in f_keys:
                if scale == 1:
                    _batches_.append(hr_batch)
                else:
                    _batches_.append(np.array(hr2lr_batch(
                        hr_batch, scale=scale, shape=lr_shape, keepdim=True)))

            return _multidata_util_(index, _batches_)

        else:
            raise ValueError(
                "When calling flow_from_h5, lr_shape should be specified.")



