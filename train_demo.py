import os
import tensorflow as tf
from src.model import EDSR_baseline
from src.preprocess import degrade_image
from src.write2tfrec import write_dst_tfrec, load_tfrecord

train_dir = "./Image/set14"
valid_dir = "./Image/set5"
AUTOTUNE = tf.data.experimental.AUTOTUNE
SCALE = 3

if not os.path.isfile("./cache/train.tfrec"):
    write_dst_tfrec(train_dir, 10, 48, "./cache/train.tfrec")
if not os.path.isfile("./cache/valid.tfrec"):
    write_dst_tfrec(valid_dir, 10, 48, "./cache/valid.tfrec")


def preprocess(hr):
    lr, hr = degrade_image(hr, SCALE, method=2, restore_shape=False)
    return lr, hr


trdst = load_tfrecord(48, "./cache/train.tfrec").map(preprocess,
                                                     AUTOTUNE).repeat()
valdst = load_tfrecord(48, "./cache/valid.tfrec").map(preprocess).repeat()

model = EDSR_baseline(scale=SCALE, model_name="EDSR_Baseline",
                      channel=3).create_model(load_weights=False,
                                              weights_path=None)

model.fit(trdst,
          valdst,
          nb_epochs=100,
          steps_per_epoch=80000 // 16,
          batch_size=16,
          use_wn=False)
