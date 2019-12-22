# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient


class LayerNet:

    def __init__(self, d, M, C):
        np.random.seed(100)
        self.params = {}
        self.params['W1'] = np.random.normal(0, 1.0/d, (d, M))
        self.params['b1'] = np.random.normal(0, 1.0/d, (M))
        self.params['W2'] = np.random.normal(0, 1.0/M, (M, C))
        self.params['b2'] = np.random.normal((C))

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x:入力データ, t:教師データ
    # def loss(self, x, t):
    #     y = self.predict(x)
        
    #     return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

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
