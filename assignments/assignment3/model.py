import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers

        self.cl1 = ConvolutionalLayer(
                                in_channels=input_shape[2],
                                out_channels=conv1_channels,
                                filter_size=3,
                                padding=1)
        self.nonlin1 = ReLULayer()
        self.mpl1 = MaxPoolingLayer(pool_size=2, stride=2)
        self.cl2 = ConvolutionalLayer(
                                    in_channels=conv1_channels,
                                    out_channels=conv2_channels,
                                    filter_size=3,
                                    padding=1)
        self.nonlin2 = ReLULayer()
        self.mpl2 = MaxPoolingLayer(pool_size=2, stride=2)
        self.flat = Flattener()
        self.fc = FullyConnectedLayer(
                                    n_input=int(input_shape[0] / 4) ** 2 * conv2_channels,
                                    n_output=n_output_classes)

        self.net = [self.cl1, self.nonlin1, self.mpl1, self.cl2, self.nonlin2,
                        self.mpl2, self.flat, self.fc]


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        for layer in self.net:
            layer.reset()

        # X_ = X.copy()
        # for layer in self.net:
        #     X_ = layer.forward(X_)

        for layer in self.net:
            X = layer.forward(X)

        # loss, grad = softmax_with_cross_entropy(X_, y)
        loss, grad = softmax_with_cross_entropy(X, y)

        # d_out = grad.copy()
        # for layer in reversed(self.net):
        #     d_out = layer.backward(d_out)

        for layer in reversed(self.net):
            grad = layer.backward(grad)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        # pred = X
        # for layer in self.net:
        #     pred = layer.forward(pred)

        # return pred.argmax(axis=1)
        pred = np.zeros(X.shape[0], np.int)
        for layer in self.net:
            X = layer.forward(X)
        pred = np.argmax(X, axis=1)
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        # raise Exception("Not implemented!")
        result = {}
        # for index, layer in enumerate(self.net):
        #     for param_name, param in layer.params().items():
        #         result[param_name + '_' + str(index)] = param
        result = {}
        for index, layer in enumerate(self.net):
            for param_name, param in layer.params().items():
                result[param_name + '_' + str(index)] = param

        return result
