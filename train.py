import matplotlib.pyplot as plt
from ast import literal_eval
from scipy import ndimage
from scipy import signal
from scipy import misc
from PIL import Image
import numpy as np
import shutil
import h5py
import sys
import os

from model import EDSR
from image_utils import hr2lr, hr2lr_batch, slice_random, hrlr_sliceFirst
from Flow import save_h5, image_flow_h5

############################################
# DATA PRE-PROCESSING
############################################

TRAIN_DIR = "../Dataset/DIV2K_train_HR/"
VALID_DIR = "../Dataset/DIV2K_valid_HR/"
H5_TRAIN = "/media/mulns/F25ABE595ABE1A75/H5File/div2k_diff_248X_train.h5"
H5_VALID = "/media/mulns/F25ABE595ABE1A75/H5File/div2k_diff_248X_valid.h5"


if not os.path.exists(H5_TRAIN):
    save_h5(image_dir=TRAIN_DIR, save_path=H5_TRAIN,
            your_func=hrlr_sliceFirst, scale=[2, 4, 8],
            slice_type=slice_random, hr_size=48, hr_stride=48, lr_shape=0,
            threshold=None, nb_blocks=1000, mode="auto")
with h5py.File(H5_TRAIN, 'r') as hf1:
    num_train = int(hf1["num_blocks"].value)
    print("Number of training blocks : ", num_train)

if not os.path.exists(H5_VALID):
    save_h5(image_dir=VALID_DIR, save_path=H5_VALID,
            your_func=hrlr_sliceFirst, scale=[2, 4, 8],
            slice_type=slice_random, hr_size=48, hr_stride=48, lr_shape=0,
            threshold=None, nb_blocks=1000, mode="auto")
with h5py.File(H5_VALID, 'r') as hf2:
    num_valid = int(hf2["num_blocks"].value)
    print("Number of validation blocks : ", num_valid)


############################################
# DATA GENERATOR
############################################

# train_gen_2X = image_flow_h5(H5_TRAIN, batch_size=16, big_batch_size=5000,
#                           keep_batch_size=False, shuffle=True, loop=True, epoch=None, index=("lr_2X", "hr"), normalize=True, reduce_mean=False)
# valid_gen_2X = image_flow_h5(H5_VALID, batch_size=100, big_batch_size=1000,
#                           keep_batch_size=False, shuffle=False, loop=True, epoch=None, index=("lr_2X", "hr"), normalize=True, reduce_mean=False)
# train_gen_4X = image_flow_h5(H5_TRAIN, batch_size=16, big_batch_size=5000,
#                           keep_batch_size=False, shuffle=True, loop=True, epoch=None, index=("lr_4X", "hr"), normalize=True, reduce_mean=False)
# valid_gen_4X = image_flow_h5(H5_VALID, batch_size=100, big_batch_size=1000,
#                           keep_batch_size=False, shuffle=False, loop=True, epoch=None, index=("lr_4X", "hr"), normalize=True, reduce_mean=False)

############################################
# TRAINING 2X MODEL
############################################

# edsr = EDSR("div2k_2X", input_size=(24, 24, 3), scale=2)
# edsr.create_model(load_weights=True)
# for lr in [5e-5,1e-5,5e-6]:
#     edsr.fit(train_gen_2X, valid_gen_2X, num_train, num_valid,
#             learning_rate=lr, nb_epochs=15, batch_size=16, loss="mae")
#     edsr.create_model(load_weights=True)

############################################
# TRAINING 4X MODEL from PRE-TRAINED 2X
############################################

# shutil.copyfile("./weights/EDSR_div2k_2X.h5", "./weights/EDSR_div2k_4X.h5")

# edsr = EDSR("div2k_4X", input_size=(12, 12, 3), scale=4)
# for lr in [5e-5, 1e-5,5e-6]:
    
#     edsr.create_model(load_weights=True)
#     edsr.fit(train_gen_4X, valid_gen_4X, num_train, num_valid,
#             learning_rate=lr, nb_epochs=15, batch_size=16, loss="mae")

############################################
### EVALUATE EDSR
############################################

edsr = EDSR("div2k_4X", input_size=(12, 12, 3), scale=4)
edsr.create_model(load_weights=True)
edsr.evaluate_batch("../Dataset/DIV2K_valid_HR", lr_shape=0, scale=4, mode="Y", verbose=1)
