import numpy as np
from mnist import MNIST
mndata = MNIST("C:/Users/masashi/DeepLearning/deeplearning-from-zero/deep-learning-from-scratch/ch03/dataset/")
# ""の中は train-images-idx3-ubyte と train-labels-idx1-ubyte を置いたディレクトリ名とすること
X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],28,28))
Y = np.array(Y)
print("hello, world")