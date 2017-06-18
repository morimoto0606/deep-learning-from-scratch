import numpy as np

def mean_squared_erro(y, t):
    sum_square = np.sum((y - t) ** 2)
    return 0.5 * sum_square

def cross_entropy_error(y, t):
    log_y = np.log(y)
    sum = np.sum(log_y * t)
    return -sum

t = np.array([0, 0, 1, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.1, 0.1, 0.05])
print(mean_squared_erro(y, t))
print(cross_entropy_error(y, t))

y = np.array([0.1, 0.1, 0.2, 0.3, 0.1, 0.3])
print(mean_squared_erro(y, t))
print(cross_entropy_error(y, t))

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_train, t_train) \
    = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)
x_batch = x_train[batch_mask]
print(x_batch)
t_batch = t_train[batch_mask]
print(t_batch)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x
def function_2(x):
    return x[0] ** 2 + x[1] ** 2

import matplotlib.pyplot as plt
x = np.arange(-3, 3, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
y = np.arange(-3, 3, 0.1)
X, Y = np.meshgrid(x, y)
a = np.array([X, Y])
z = function_2(a)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X, Y, z)
plt.show()

def numerical_diff(f, x):
    h = 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

print(numerical_diff(function_1, 10), 2 * 10 * 0.01 + 0.1)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

#grad = numerical_gradient(function_2, a)
#plt.quiver(X, Y, grad)
def gradient_descent(f, init_x, l=0.01, step_num = 100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= l * grad
    return x

print(gradient_descent(function_2, np.array([-1.0, -1.0])))
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, 0.01))
