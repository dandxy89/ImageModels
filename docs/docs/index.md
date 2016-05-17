### Motivation

Learn to build and experiment with well-known Image Processing Neural Network Models. As yet, there is no intention to train or run the models. This repository only served as a learning exercise to understand how these models are built and how to use the new Keras Functional API.

The second reason is that I do not have enough computing resource to fully train such a models ( :( )with the computing resource I have currently available.... one day!

!!! warning
    Models configuration may differ slightly from the original implementation

### Intention

The purpose of the repository is to re-create ImageNet winners in Keras and to utilize their pretrainied model weights. The folllowing models will be recreated:

1.  AlexNet
    *   Caffe version
    *   Added 9/5/2016
    *   *Alexnet.py*
2.  AlexNet
    *   From Original Paper Diagram
    *   Added 9/5/2016
    *   *AlexNet_Unmodified.py*
3.  CaffeNet
    *   Caffe version
    *   Added 11/5/2016
    *   *CaffeNet.py*
4.  VGG-19
    *   Caffe Version
    *   Added 9/5/2016
    *   *VGG-19.py*
5.  GoogLeNet
    *   As described in Original paper
    *   Added 11/5/2016

The next intention is to check if I can setup Tensorflow distributed training and Tensorflow Serving, untrained models will be used for this.

### Dependencies

*   Keras
*   Theano / Tensorflow
*   Matplotlib
*   Numpy
*   Pydot

### Directory Structure

    /ImageNetModels
        /KerasLayers
            Custom_Layers.py - Contains LRN2D layer
        /Model
            *.png - pydot visulisations of each of the models
            *.txt - outputs from building the models
        /Serving - reserved for the exported Tensorflow models
        /docs - MKDocs files
        *.py - Keras Models
