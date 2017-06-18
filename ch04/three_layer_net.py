import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient
import numpy as np

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size1, hidden_size2,
                 output_size, weight_int_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_int_std \
                            * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size2)
        self.params['W2'] = weight_int_std \
                            * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)

        self.params['W3'] = weight_int_std \
                            * np.random.randn(hidden_size2, output_size)
        self.params['b3'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b2, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    #x: input t: teacher
    def numerical_gradient(self, x, t):
        loss_w = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_w, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_w, self.params['b3'])
        return grads


m = np.array([[1, 2, 3], [4, 5, 6]])
x = np.array([[1], [3], [2]])
y = np.dot(m, x)
print(m, x, y)

net = ThreeLayerNet(input_size=784, hidden_size1=100, hidden_size2=299, output_size=10)
print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)
print(net.params['W3'].shape)
print(net.params['b3'].shape)
