# SuperSR 🎈

An implementation of data-pipeline and models of SR (super-resolution based on CNN) research.

## Updates 19.6.18

Amazing, it has been year 9012 yet I'm still here super-resolving 2D images... **Remember to update tensorflow2.0!**

Details about the usage of ***SuperSR***, check the notebooks: [train_model](/train_models.ipynb "notebook") and [train_customized_model](/train_customized_model.ipynb "notebook")

### Brand new Codes

- Faster
- More Stable
- More Concise
- Easy to read and use
(As far as I'm thinking...)
- Tutorial prepared

### Main Structure

- ***Model establishing,***

  - Thanks to `tf.keras`, we can build model really really fast and easy.

  - With `BaseSRModel`, everything about training or evaluation(to be implemented) a super-resolution model could be finished in one time. All you need to do is to customize the model structure part in the method named `create_model`.
  
  - Here uses Adam as default optimizer, you can change it in `BaseSRModel`.

  - I add weight-normalization for Adam optimizer, one can set `use_wn` True to use it.

  - It's noted that the `lr_schedule` method is the most common schedule solution of learning rate in my training. One can modify it anyway, such as `SRCNN` model (original paper has defined a learning rate schedule), it's flexible~

  - Pre-defined models, such as `EDSR`, `SRCNN`, are ready to be trained directly. (Basically follow the original paper.)

- ***Data pipeline,***

  - Thanks to `tf.data`, we can easily build tensorflow dataset and use `tfrecord` file to accelerate pipeline when training with `tf.keras`.

  - One need to crop training images into patches which is usually done in super-resolution research. Here we strongly suggest u to write hr-patches into `tfrecord` files with `write_dst_tfrec` function, since loading image files into models will cost I/O time consumption. *Pre-writing patches into tfrecord file -> loading them -> mapping with preprocess function* is the best way for u to save time for :coffee

  - I pre-defined many preprocessing functions such as `downsample_gaussian` (downsample image with gaussian kernel) and `degrade_image` (degrade image with blur kernel and noise). One can use them directly, the documentations are comprehensive and clear.

  - Remember ***DO NOT*** batch the dataset before feeding into the Model, because the `fit` function of `BaseSRModel` will batch it based on the batch size you set.

### Future Work

- [ ] More pre-defined model such as WDSR, VDSR and so on...
- [ ] Evaluation part of trained model. It's a big work...
- [x] Add notebooks for instruction of training with `**SuperSR**` !
- [ ] Train some model weights and share to everyone~

----

Mulns,
May world peace prevail...
mulns@outlook.com
