# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def init_network2d(I, d, M, C):
    np.random.seed(100)
    network = {}
    network["W1"] = np.random.normal(0, 1.0/d, (M, d))
    network["b1"] = np.random.normal(0, 1.0/d, (M))
    network["W2"] = np.random.normal(0, 1.0/M, (C, M))
    network["b2"] = np.random.normal(0, 1.0/M, (C))
    return network

if __name__ == "__main__":
    print("画像の選択...")
    id = int(input())
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    network = init_network2d(10000, 784, 20, 10)
    W1 = network["W1"]
    b1 = network["b1"]
    W2 = network["W2"]
    b2 = network["b2"]
    # 入力層の出力
    y0 = x_train[id]
    print(y0)
    # 中間層の出力
    y1 = sigmoid(np.dot(W1, y0) + b1)
    # 出力層の出力
    y2 = softmax(np.dot(W2, y1) + b2)
    # print(y2)
    print(np.argmax(y2))
