### Info

Only one version of CaffeNet has been built

```
@article{ding2014theano,
  title={Theano-based Large-Scale Visual Recognition with Multiple GPUs},
  author={Ding, Weiguang and Wang, Ruoyan and Mao, Fei and Taylor, Graham},
  journal={arXiv preprint arXiv:1412.2302},
  year={2014}
}
```

### Keras Model Visulisation

**CaffeNet**

![CaffeNet](Images/CaffeNet.png)

### Keras Model Builds

    # Channel 1 - Convolution Net Input Layer
    x = conv2D_lrn2d(
        img_input, 3, 11, 11, subsample=(
            1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 1
    x = conv2D_lrn2d(x, 96, 55, 55, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = LRN2D(alpha=ALPHA, beta=BETA)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 2
    x = conv2D_lrn2d(x, 192, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = LRN2D(alpha=ALPHA, beta=BETA)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 3
    x = conv2D_lrn2d(x, 288, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 4
    x = conv2D_lrn2d(x, 288, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 5
    x = conv2D_lrn2d(x, 256, 13, 13, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Cov Net Layer 7
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Channel 1 - Cov Net Layer 8
    x = Dense(4096, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Final Channel - Cov Net 9
    x = Dense(output_dim=NB_CLASS,
              activation='softmax')(x)