"""
    Model Name:

        AlexNet - using the Functional Keras API

    Paper:

         ImageNet classification with deep convolutional neural networks by Krizhevsky et al. in NIPS 2012

    Alternative Example:

        Available at: http://caffe.berkeleyvision.org/model_zoo.html

        https://github.com/uoguelph-mlrg/theano_alexnet/tree/master/pretrained/alexnet

    Original Dataset:

        ILSVRC 2012

"""
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras import regularizers
from keras.utils.visualize_util import plot


# global constants
NB_CLASS = 1000         # number of classes
DROPOUT = 0.4
WEIGHT_DECAY = 0.0005   # L2 regularization factor
USE_BN = True           # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
DIM_ORDERING = 'th'


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


def conv_layer(x, nb_filter, nb_row, nb_col, dim_ordering,
               subsample=(1, 1), activation='relu',
               border_mode='same', weight_decay=None, padding=None):

    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      bias=False,
                      dim_ordering=dim_ordering)(x)

    if padding:
        for i in range(padding):
            x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    return x


def create_model():
    # Define image input layer
    if DIM_ORDERING == 'th':
        INP_SHAPE = (3, 224, 224)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        INP_SHAPE = (224, 224, 3)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

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

    return x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING


def check_print():
    # Create the Model
    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

    # Create a Keras Model - Functional API
    model = Model(input=img_input,
                  output=[x])
    model.summary()

    # Save a PNG of the Model Build
    plot(model, to_file='./Model/GoogleNet.png')

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')
    print('Model Compiled')
