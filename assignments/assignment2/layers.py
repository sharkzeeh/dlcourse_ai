import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    # implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W * W)
    grad = 2 * reg_strength*W
    return loss, grad


# used for softmax_with_cross_entropy function
def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
        predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
        probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # implement softmax
    # Your final implementation shouldn't have any loops
    # print(predictions.shape)
    preds = predictions.copy()

    if predictions.ndim == 1:
        preds -= np.max(preds)
        expons = np.exp(preds)
        probs = expons / np.sum(expons)

    else:
        preds -= np.max(preds, axis = 1).reshape(-1, 1)
        expons = np.exp(preds)
        probs = expons / expons.sum(axis = 1).reshape(-1, 1)

    return probs


# used for softmax_with_cross_entropy
def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
        probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class

        target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
        loss: single value
    '''
    # implement cross-entropy
    # Your final implementation shouldn't have any loops
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])

    else:
        loss = (-np.log(probs[np.arange(probs.shape[0]), target_index.flatten()])).mean()

    return loss

def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment

    # implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops

    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()

    dpr_len = dprediction.shape[0]

    if dprediction.ndim == 1:
        dprediction[target_index] -= 1

    else:
        dprediction[np.arange(dpr_len), target_index.flatten()] -= 1
        dprediction /= dpr_len

    return loss, dprediction



class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

    def reset(self):
        """
        Resets accumulated gradient from previous run
        """
        self.grad = np.zeros_like(self.value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass

        self.dZdX = X > 0
        return np.maximum(X, 0)

    def backward(self, dLdZ):
        """
        Backward pass
        Arguments:
        dLdZ, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        d_result = dLdZ * self.dZdX #  нужно просто заменить в градиенте значения < 0 на 0
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return X @ self.W.value + self.B.value


    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_result = d_out @ self.W.value.T

        dLdW = self.X.T @ d_out
        dLdB = np.sum(d_out, axis = 0) # ох уж эти оси: сложение соотв. элементов в строках (друг под другом)
        self.W.grad += dLdW
        self.B.grad += dLdB

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
