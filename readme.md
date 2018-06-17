# SuperSR

A completion of data-preprocessing and models of SR(super-resolution based on CNN) research.

## Image_utils

- Image downsampling, supports batch operation.
- Image slicing, supports random, normal and remove_redundance modes. Remove redundance.
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
- [ ]    Add texture extraction method in Dataset.
- [ ]    Add clustering algorithm in utils.py
- [ ]    try different upsample module in EDSR 8X model.
- [ ]    Figure out how to import my own moduals.
- [x]    ~~Rewrite the image_utils.py to be more readable and scalable.~~
- [ ]    Add feature extraction in image_utils.py
- [ ]    Fix Comments in image_utils.py
- [ ]    Test if the models work on this version.
- [ ]    Finish saving image into directory.
- [ ]    Finish image flow from directory.