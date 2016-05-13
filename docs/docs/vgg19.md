### Info

Only one version of VGG-19 has been built

```
@article{DBLP:journals/corr/SimonyanZ14a,
      author    = {Karen Simonyan and
                   Andrew Zisserman},
      title     = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
      journal   = {CoRR},
      volume    = {abs/1409.1556},
      year      = {2014},
      url       = {http://arxiv.org/abs/1409.1556},
      timestamp = {Wed, 01 Oct 2014 15:00:05 +0200},
      biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/SimonyanZ14a},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
```

### Keras Model Visulisation

**VGG-19**

![VGG-19](Images/VGG19.png)

### Keras Model Builds

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