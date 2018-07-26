# SuperSR

A completion of data-preprocessing and models of SR(super-resolution based on CNN) research.

## Updates 7.13

*Add FlowData Class!*

- Feature:
  - Highly scalable, U can define all functions you need to generate different kind of data.
  - Easy to use, only need to implement one or two function. SR commonly used class has implemented.
- Usage:
  - Inherent from FlowData class (See SRFlowData Example) and implement two function.
    - The `before_save` function is used to generate blocks before saving blocks into h5 file or directory. Input will be a String defines the path to a single image. You will slice the image into blocks (or multi kind of blocks) and return a dictionary. `key` will be regarded as the name of Dataset in h5 file or name of directory. `value` should be a numpy array. If save to directory, it also need to be Int8 dtype.
    - The `before_flow` function is used to preprocess data flow from h5 file. Input will be a List of batches. Each elements in batches is a numpy array in shape of (N, h, w [,c]), N defines the number of blocks, c is the channel number which only when bigger than 1. You will process the batch of blocks and return a tuple if you want to fit into keras models. (Of course you can define any data-type to flow, it will return just as what you defined.)

*Other updates...*

- Enhance the models' fitting mode, supports multi-inputs and multi-outputs situation. Also supports the multi-stage when fitting with different learning_rate and epochs.
- Enhance the `utils.psnr` and `utils.PSNR` function, no need to concern the peak-value of image at all to calculate psnr value. (But still only supports 8-bit image.)

## Image_utils

- Image downsampling, supports batch operation.
- Image slicing, supports random, normal and remove_redundance modes.
- Merge image from blocks, supports multichannels.
- Image preprocessing in downsample first(not work yet..) and slicing first(works fine) mode.
- Image generator, one image a time, supports customize processing function.
- Save to h5 file, supports big data.
- Data flow from h5 file, supports big data.

## Models

- Base model class, the base class of all models, with train and evaluation func.
- SRCNN model
- Residual-Network, supports sub-pixel upscaling.
- EDSR, using sub-pixel upsaling instead of deconv layer, and using mae instead of mse to evaluate loss.

## To Do List

- [ ]    Multiprocessing when generate h5 files and or subimages' dir
- [ ]    Try PyTables to save and generate data.
- [x]    ~~Finish the downsample first mode when generating data and label.~~
- [ ]    Comments are incomplete, under fixing.
- [x]    ~~Delete the slice first mode and set downsample first mode as Default.~~
- [ ]    Tensorboard visulization is incompleted. Weights visualization need to be considered.
- [x]    ~~Add texture extraction method in Dataset.~~
- [x]    ~~Add clustering algorithm in utils.py~~
- [x]    ~~try different upsample module in EDSR 8X model.~~
- [x]    ~~Figure out how to import my own moduals.~~
- [x]    ~~~~Rewrite the image_utils.py to be more readable and scalable.~~~~
- [x]    ~~Add feature extraction in image_utils.py~~
- [x]    ~~Test if the models work on this version.~~
- [x]    ~~Change flow function to adjust diff pre-processing.~~
- [x]    ~~Adjust model.fit function to multi-stages.~~
- [x]    ~~Adjust index param in flow to multi-inputs/outputs in keras.~~
- [ ]    Finish saving image into directory.
- [ ]    Finish image flow from directory.
- [ ]    Add visualization of different layers in models.