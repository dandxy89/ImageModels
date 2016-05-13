"""
    Model Name:

        VGG-19 - using the Functional Keras API

        A model of the 19-layer network used by the VGG team in the ILSVRC-2014 competition.

    Paper:

         Very Deep Convolutional Networks for Large-Scale Image Recognition - K. Simonyan, A. Zisserman

         arXiv:1409.1556

    Alternative Example:

        Available at (Caffe) : http://caffe.berkeleyvision.org/model_zoo.html

    Original Dataset:

        ILSVRC 2012

"""
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras.utils.visualize_util import plot


def create_model():
    # Global Constants
    NB_CLASS = 1000  # number of classes
    # 'th' (channels, width, height) or 'tf' (width, height, channels)
    DIM_ORDERING = 'th'

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

    # Layer Cluster - 1
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(img_input)
    x = Convolution2D(64, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(64, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Layer Cluster - 2
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(128, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(128, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Layer Cluster - 3
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(256, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(256, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(256, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(256, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Layer Cluster - 4
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Convolution2D(512, 3, 3,
                      activation='relu',
                      border_mode='same',
                      dim_ordering=DIM_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Layer Cluster - 5 - Output Layer
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='softmax')(x)

    return x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING


def check_print():
    # Create the Model
    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

    # Create a Keras Model - Functional API
    model = Model(input=img_input,
                  output=[x])
    model.summary()

    # Save a PNG of the Model Build
    plot(model, to_file='./Model/VGG19.png')

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')
    print('Model Compiled')
