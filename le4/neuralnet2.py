# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.functions import sigmoid, softmax
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def init_network3d(B, d, M, C):
    np.random.seed(100)
    network = {}
    network["W1"] = np.random.normal(0, 1.0/d, (B, M, d))
    network["b1"] = np.random.normal(0, 1.0/d, (B, M))
    network["W2"] = np.random.normal(0, 1.0/M, (B, C, M))
    network["b2"] = np.random.normal(0, 1.0/M, (B, C))
    return network

# def forward():
    

if __name__ == "__main__":

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=True)

    # 中間層のノード数
    M = 20
    # バッチサイズ
    B = 100

    # ネットワーク初期化
    network = init_network3d(100, 784, 20, 10)
    W1 = network["W1"]
    b1 = network["b1"]
    W2 = network["W2"]
    b2 = network["b2"]

    # 入力層の出力
    choice = np.random.choice(range(0, 59999), B)
    y0 = x_train[choice]    # (100, 784)
    print(y0.shape)

    # 中間層の出力
    y1 = np.array([[0]*M])
    for i in range(0, B):
        y1 = np.append(y1, [np.dot(W1[i], y0[i])], axis=0)
    y1 = np.delete(y1, 0, axis=0) # (100, 20)
    y1 = sigmoid(y1 + b1)
    print(y1.shape)
    
    # 出力層の出力
    y2 = np.array([[0]*10])
    for i in range(0, B):
        y2 = np.append(y2, [np.dot(W2[i], y1[i])], axis=0)
    y2 = np.delete(y2, 0, axis=0) # (100, 10)
    y2 = softmax(y2 + b2)
    print(y2.shape)

    # クロスエントロピー誤差の計算
    cross_entropy_sum = 0
    for i in range(0, B):
        logy = np.log(y2[i]+1e-9)
        cross_entropy_sum += np.sum(-logy * t_train[choice[i]])
    print(cross_entropy_sum / B)