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

def grad_cross_entropy(Yk2, Yk, B): # Yk2 => (B * C), Yk => (B * C)
	return (Yk2 - Yk) / B

def grad_fully_layer(W, X, dEn_dY):
	grad = {}
	Wt = W.T
	Xt = X.T
	grad["X"] = np.dot(dEn_dY, Wt)
	print(Wt.shape)
	print(dEn_dY.shape)
	print(grad["X"].shape)

	print(dEn_dY.shape)
	print(Xt.shape)
	grad["W"] = np.dot(Xt, dEn_dY)
	grad["b"] = np.sum(dEn_dY, axis=0)
	return grad

def grad_sigmoid(B, M, Y, dEn_dY):
	I = np.full((B, M), 1)
	print(dEn_dY.shape)
	print(Y.shape)
	return dEn_dY * (I - Y) * Y


if __name__ == "__main__":

	(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=True)

	# 中間層のノード数
	M = 20
	# バッチサイズ
	B = 100
	# 学習率
	eta = 0.01

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
	x1 = np.array([[0]*M])
	for i in range(0, B):
		x1 = np.append(x1, [np.dot(W1[i], y0[i])], axis=0)
	x1 = np.delete(x1, 0, axis=0) # (100, 20)
	x1 = x1 + b1
	y1 = sigmoid(x1)
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
	
	# ソフトマックスの偏微分の計算
	dEn_dak = grad_cross_entropy(y2, t_train[choice], B)
	print("softmax")
	print(dEn_dak.shape)

	# 出力層の偏微分の計算
	grad2 = grad_fully_layer(W2, y1, dEn_dak)
	dEn_dW2 = grad2["W"]
	dEn_db2 = grad2["b"]
	dEn_dX2 = grad2["X"]

	# シグモイドの偏微分の計算
	dEn_dxs = grad_sigmoid(B, M, x1, dEn_dX2)

	# 中間層の偏微分の計算
	gra1 = grad_fully_layer(W1, y0, dEn_dxs)
	dEn_dW1 = grad2["W"]
	dEn_db1 = grad2["b"]
	dEn_dX1 = grad2["X"]

	# パラメータの更新
	W1 = W1 - eta * dEn_dW1
	b1 = b1 - eta * dEn_db1
	W2 = W2 - eta * dEn_dW2
	b2 = b2 - eta * dEn_db2
	print(dEn_dW1)