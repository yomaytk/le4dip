import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import warnings
warnings.filterwarnings('ignore')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def init_network2d(I, d, M, C):
    np.random.seed(100)
    network = {}
    network["W1"] = np.load('W1.npy')
    network["b1"] = np.load('b1.npy')
    network["W2"] = np.load('W2.npy')
    network["b2"] = np.load('b2.npy')
    return network

if __name__ == "__main__":
    print("画像の選択...")
    id = int(input())
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    network = init_network2d(10000, 784, 100, 10)
    W1 = network["W1"]
    b1 = network["b1"]
    W2 = network["W2"]
    b2 = network["b2"]
    # 入力層の出力
    y0 = x_train[id]
    # 中間層の出力
    y1 = sigmoid(np.dot(W1.T, y0) + b1)
    # 出力層の出力
    y2 = softmax(np.dot(W2.T, y1) + b2)
    # print(y2)
    print(np.argmax(y2))
    print(y2)
