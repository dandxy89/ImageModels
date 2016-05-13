"""
    My Current Environment has been setup to use: Theano

    Script provides a number of functions:

        Demonstrates that the Compile Model can process an Output

        Demonstrates who to export the Tensorflow Model (Issue)

        Demonstrates how to visualise slices of the Convolutional Layers

        Saves model summary and Image to the Model folder

    Created 13/5/2016
"""
import numpy as np
from keras.models import Model

from AlexNet import check_print as AlexNetModelcheck_print
from AlexNet import create_model as AlexNetModel
from AlexNet_Original import check_print as AlexNetOrigModelcheck_print
from AlexNet_Original import create_model as AlexNetOrigModel
from CaffeNet import check_print as CaffeNetModelcheck_print
from CaffeNet import create_model as CaffeNetModel
from GoogLeNet import check_print as GoogleNetModelcheck_print
from GoogLeNet import create_model as GoogleNetModel
from VGG19 import check_print as VGG19Modelcheck_print
from VGG19 import create_model as VGG19Model

model_choice = dict(AlexNet=AlexNetModel,
                    AlexNetOrig=AlexNetOrigModel,
                    CaffeNet=CaffeNetModel,
                    GoogLeNet=GoogleNetModel,
                    VGG19=VGG19Model)


model_val = 'AlexNet'
test_batch = False
get_graph = False
show_activation = True
show_cmd_output = False


if not show_cmd_output:
    """
        Compiles the respective Model of choice

    """

    print('Building : {}'.format(model_val))
    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = model_choice[
        model_val]()

    # Create a Keras Model - Functional API
    model = Model(input=img_input,
                  output=[x])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')

if test_batch:
    """
        Setup:

            test_batch = True
            get_graph = False
            show_activation = False
            show_cmd_output = False

    """

    print('Testing Model Output with Dummy Data...')

    # Check that it can predict...
    batch_n = 5  # Test Batch Size
    test_images = np.random.rand(batch_n, 3, 224, 224)

    # Get Model Prediction
    output = model.predict(x=test_images,
                           batch_size=batch_n)

    # Get the Softmax Prediction per Image
    print(np.argmax(output, axis=1))

if get_graph:
    """
        Having some difficulty with the installation of Tensorflow Serving on Mac!

        AHH!

        Setup:

            test_batch = False
            get_graph = True
            show_activation = False
            show_cmd_output = False

    """

    from keras import backend as K

    # all new operations will be in test mode from now on
    K.set_learning_phase(0)

    # serialize the model and get its weights, for quick re-building
    config = model.get_config()
    weights = model.get_weights()

    # re-build a model where the learning phase is now hard-coded to 0
    from keras.models import model_from_config

    new_model = model_from_config(config)
    new_model.set_weights(weights)

    import tensorflow as tf
    import sys
    sys.path.insert(0, '/Users/dan.dixey/Desktop/QBiz/serving')
    # Unable to Import THIS!! why?
    from tensorflow_serving.session_bundle import exporter

    sess = K.get_session()

    export_path = './Serving'  # where to save the exported graph
    export_version = 0o0000001  # version number (integer)

    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    signature = exporter.classification_signature(input_tensor=model.input,
                                                  scores_tensor=model.output)
    model_exporter.init(sess.graph.as_graph_def(),
                        default_graph_signature=signature)

    model_exporter.export(export_path, tf.constant(export_version), sess)

if show_activation:
    """
        Experimenting with different the Cov Net Layers to visulise their outputs

        This section will push data through the layer and record the activations

        Normalise (Min/Max) the Data in the whole 3D array, slice for one layer of the 3D Array

        Plot using Maplotlib the Output of 10 slice...

        Setup:

            test_batch = False
            get_graph = False
            show_activation = True
            show_cmd_output = False

    """

    from keras import backend as K
    import matplotlib.pyplot as plt
    from time import sleep

    n_layers = len(model.layers)

    print('Number of Layers to choose from: {}'.format(n_layers))
    zip(model.layers, range(n_layers))
    layer_n = input('Enter a Number for the Layer to Viz: ')

    def get_activations(model, layer, X_batch):
        # Credit for this function belongs to *damaha* (Github Username)
        # https://github.com/fchollet/keras/issues/41
        # from keras import backend as K

        get_activations = K.function(
            [model.layers[0].input, K.learning_phase()], model.layers[layer].output)
        activations = get_activations([X_batch, 0])

        return activations

    # Check that it can predict...
    batch_n = 5  # Test Batch Size
    test_images = np.random.rand(batch_n, 3, 224, 224)

    layer_output = get_activations(
        model=model,
        layer=layer_n,
        X_batch=test_images)

    # Visualisation
    def normalise_3d(data):
        min, max = data.min(), data.max()
        return (data - min) / (max - min)

    data = normalise_3d(data=layer_output[1, :, :, :])

    for slice_v in range(10):

        H = data[slice_v, :, :]

        fig = plt.figure(figsize=(6, 3.2))

        ax = fig.add_subplot(111)
        ax.set_title('Visualise slice {} within the Layer'.format(slice_v))
        plt.imshow(H)
        ax.set_aspect('equal')

        cax = fig.add_axes([0, 1, 0, 1])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
        plt.show()

        sleep(2)

if show_cmd_output:
    """
        Save the Command Line Output to a text file

        Example: python Train.py > Model/AlexNet.txt

        Setup:

            test_batch = False
            get_graph = False
            show_activation = False
            show_cmd_output = True

    """

    print('Model : {}'.format(model_val))

    model_print = dict(AlexNet=AlexNetModelcheck_print,
                       AlexNetOrig=AlexNetOrigModelcheck_print,
                       CaffeNet=CaffeNetModelcheck_print,
                       GoogLeNet=GoogleNetModelcheck_print,
                       VGG19=VGG19Modelcheck_print)

    # Run Function to get Output
    model_print[model_val]()
