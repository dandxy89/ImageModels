### Info

Only one version of CaffeNet has been built

```
Going deeper with convolutions
Szegedy, Christian; Liu, Wei; Jia, Yangqing; Sermanet, Pierre; Reed, Scott; Anguelov, Dragomir;
Erhan, Dumitru; Vanhoucke, Vincent; Rabinovich, Andrew
arXiv:1409.4842
```

### Keras Model Visulisation

**GoogLeNet**

![GoogLeNet](Images/GoogLeNet.png)

### Keras Model Builds

**Inception**

    def inception_module(x, params, dim_ordering, concat_axis,
                         subsample=(1, 1), activation='relu',
                         border_mode='same', weight_decay=None):

        # https://gist.github.com/nervanazoo/2e5be01095e935e90dd8  #
        # file-googlenet_neon-py

        (branch1, branch2, branch3, branch4) = params

        if weight_decay:
            W_regularizer = regularizers.l2(weight_decay)
            b_regularizer = regularizers.l2(weight_decay)
        else:
            W_regularizer = None
            b_regularizer = None

        pathway1 = Convolution2D(branch1[0], 1, 1,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(x)

        pathway2 = Convolution2D(branch2[0], 1, 1,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(x)
        pathway2 = Convolution2D(branch2[1], 3, 3,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(pathway2)

        pathway3 = Convolution2D(branch3[0], 1, 1,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(x)
        pathway3 = Convolution2D(branch3[1], 5, 5,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(pathway3)

        pathway4 = MaxPooling2D(pool_size=(1, 1), dim_ordering=DIM_ORDERING)(x)
        pathway4 = Convolution2D(branch4[0], 1, 1,
                                 subsample=subsample,
                                 activation=activation,
                                 border_mode=border_mode,
                                 W_regularizer=W_regularizer,
                                 b_regularizer=b_regularizer,
                                 bias=False,
                                 dim_ordering=dim_ordering)(pathway4)

        return merge([pathway1, pathway2, pathway3, pathway4],
                     mode='concat', concat_axis=concat_axis)

**Model**

    x = conv_layer(img_input, nb_col=7, nb_filter=64,
                   nb_row=7, dim_ordering=DIM_ORDERING, padding=3)
    x = MaxPooling2D(
        strides=(
            3, 3), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)

    x = conv_layer(x, nb_col=1, nb_filter=64,
                   nb_row=1, dim_ordering=DIM_ORDERING)
    x = conv_layer(x, nb_col=3, nb_filter=192,
                   nb_row=3, dim_ordering=DIM_ORDERING, padding=1)
    x = MaxPooling2D(
        strides=(
            3, 3), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)

    x = inception_module(x, params=[(64, ), (96, 128), (16, 32), (32, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)

    x = MaxPooling2D(
        strides=(
            1, 1), pool_size=(
                1, 1), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    # AUX 1 - Branch HERE
    x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64, )],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    # AUX 2 - Branch HERE
    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = MaxPooling2D(
        strides=(
            1, 1), pool_size=(
                1, 1), dim_ordering=DIM_ORDERING)(x)

    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)],
                         dim_ordering=DIM_ORDERING, concat_axis=CONCAT_AXIS)
    x = AveragePooling2D(strides=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Flatten()(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(output_dim=NB_CLASS,
              activation='linear')(x)
    x = Dense(output_dim=NB_CLASS,
              activation='softmax')(x)