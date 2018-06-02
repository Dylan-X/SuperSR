# SuperSR

A completion of data-preprocessing and models of SR(based on CNN) research.

## To Do List

    Multiprocessing when generate h5 files and or subimages' dir

    ** Fix a little bug in Dataset class: if h5 file doesn't exist, it will regard it as a ValueError. **

    Try PyTables to save and generate data.

    ** Finish the downsample first mode when generating data and label. **

    Comments are incomplete, stucture is under enhanced.

    ** Delete the slice first mode and set downsample first mode as Default. **

    Tensorboard visulization is incompleted. Weights visualization need to be considered. 

    Add texture extraction method in Dataset.

    Add clustering algorithm in utils.py

    try different upsample module in EDSR 8X model 