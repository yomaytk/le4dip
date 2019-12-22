import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from common.functions import *
# from layer import LayerNet

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)

    B = y.shape[0]
    return -np.sum(np.log(y[np.arange(B), t] + 1e-7)) / B

def network_init(d, M, C):
    np.random.seed(100)
    params = {}
    params['W1'] = np.random.normal(0, 1.0/d, (d, M))
    params['b1'] = np.random.normal(0, 1.0/d, (M))
    params['W2'] = np.random.normal(0, 1.0/M, (M, C))
    params['b2'] = np.random.normal((C))
    return params

def predict(params, x):
    W1 = params['W1'] 
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)
    return y

def cross_loss(params, x, t):
    y = predict(params, x)
    return cross_entropy_error(y, t)

def gradient(params, x, t):
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']

    grads = {}
    
    B = x.shape[0]
    
    # forward
    # 中間層
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    # 出力層
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)
    
    # backward
    # ソフトマックス関数、クロスエントロピーの偏微分
    dy = (y - t) / B
    # 出力層の偏微分
    grads['W2'] = np.dot(z1.T, dy)
    grads['b2'] = np.sum(dy, axis=0)
    # 中間層の偏微分
    dz1 = np.dot(dy, W2.T)
    # シグモイド関数の偏微分
    da1 = sigmoid_grad(a1) * dz1
    # 中間層の偏微分
    grads['W1'] = np.dot(x.T, da1)
    grads['b1'] = np.sum(da1, axis=0)

    return grads


if __name__ == "__main__":
    
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 各層のノード数
    d = 784
    M = 100
    C = 10

    # ネットワーク初期化
    params = network_init(d, M, C)

    epoch = 10000
    train_size = x_train.shape[0]

    # バッチサイズ
    B = 100
    # 学習率
    eta = 0.1

    per_epoch = max(train_size / B, 1)

    for i in range(epoch):

        choice = np.random.choice(train_size, B)
        x_batch = x_train[choice]
        t_batch = t_train[choice]
        
        grad = gradient(params, x_batch, t_batch)
        
        params['W1'] -= eta * grad['W1']
        params['b1'] -= eta * grad['b1']
        params['W2'] -= eta * grad['W2']
        params['b2'] -= eta * grad['b2']
        
        loss = cross_loss(params, x_batch, t_batch)

        if i % per_epoch == 0:
            print(str(loss))

    np.save("W1", params['W1'])
    np.save("b1", params['b1'])
    np.save("W2", params['W2'])
    np.save("b2", params['b2'])