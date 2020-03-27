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
    preds = predictions.copy()

    if predictions.ndim == 1:
        preds -= np.max(preds)
        expons = np.exp(preds)
        probs = expons / np.sum(expons)

    else:
        preds -= np.max(preds, axis=1).reshape(-1, 1)
        expons = np.exp(preds)
        probs = expons / expons.sum(axis=1).reshape(-1, 1)

    return probs


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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

    # def reset(self):
    
    #     self.grad = np.zeros_like(self.value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
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
        d_result = dLdZ * self.dZdX
        return d_result

    def reset(self):
        """
        Resets accumulated gradient from previous run
        """
        pass

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
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
        d_result = d_out @ self.W.value.T

        dLdW = self.X.T @ d_out
        dLdB = np.sum(d_out, axis=0)
        self.W.grad += dLdW
        self.B.grad += dLdB
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}

    def reset(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(np.random.randn(
                                        filter_size,
                                        filter_size,
                                        in_channels,
                                        out_channels))

        self.B = Param(np.zeros(out_channels))
        self.padding = padding
        self.X = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = 0
        out_width = 0
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        X_padded = np.zeros((batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels))
        X_padded[:, self.padding: self.padding + height, self.padding: self.padding + width, :] = X
        self.X = (X, X_padded)
        X_padded = X_padded[:, :, :, :, np.newaxis]

        W = self.W.value[np.newaxis, :, :, :, :]

        out_height = 1 + height - self.filter_size + 2 * self.padding
        out_width = 1 + width - self.filter_size + 2 * self.padding
        res = np.zeros((batch_size, out_height, out_width, self.out_channels))

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X_padded[:, y: y + self.filter_size, x:x + self.filter_size, :, :]
                res[:, y, x, :] = np.sum(X_slice * self.W.value, axis=(1, 2, 3)) + self.B.value
        return res


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        X, X_padded = self.X
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        X_grad = np.zeros_like(X_padded)

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                X_slice = X_padded[:, y: y + self.filter_size, x: x + self.filter_size, :, np.newaxis]
                grad = d_out[:, y, x, np.newaxis, np.newaxis, np.newaxis, :]
                self.W.grad += np.sum(grad * X_slice, axis=0)

                X_grad[:, y:y + self.filter_size, x: x + self.filter_size, :] += np.sum(self.W.value * grad, axis=-1)

        self.B.grad += np.sum(d_out, axis=(0, 1, 2))

        return X_grad[:, self.padding:self.padding + height, self.padding:self.padding + width, :]

    def reset(self):
        self.W.grad, self.B.grad = np.zeros_like(self.W.value), np.zeros_like(self.B.value)

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X.copy()

        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        out = np.zeros((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range(out_width):
                X_slice = X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                out[:, y, x, :] = np.amax(X_slice, axis=(1, 2))

        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1
        out = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                X_slice = self.X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                grad = d_out[:, y, x, :][:, np.newaxis, np.newaxis, :]
                mask = (X_slice == np.amax(X_slice, (1, 2))[:, np.newaxis, np.newaxis, :])
                out[:, y:y + self.pool_size, x:x + self.pool_size, :] += grad * mask
        return out

    def reset(self):
        pass

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def reset(self):
        pass

    def params(self):
        # No params!
        return {}
