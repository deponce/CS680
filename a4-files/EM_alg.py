import numpy as np
from model.models import *
import gzip
import matplotlib.pyplot as plt
import time
gmm_dataset = np.loadtxt("./data/gmm_dataset.csv", delimiter=",")
loss_1_10=[]
k_s = range(3,11)
for k in k_s:
    model = GMM(k=k, max_iter=500)
    loss = model.fit(gmm_dataset)
    loss_1_10.append(loss[-1])
plt.plot(k_s, loss_1_10)
plt.show()

def get_x_y(mnist):
    y = mnist[:,0]
    x = mnist[:,1:]
    return x, y


print("load Mnist dataset")
mnist_training = np.loadtxt("./data/mnist_train.csv", delimiter=',')
mnist_test = np.loadtxt("./data/mnist_test.csv", delimiter=',')
mnist_test_y = mnist_test[:,0]
mnist_test_x = mnist_test[:,1:].reshape((10000,28,28))
mnist_training_y = mnist_training[:,0]
mnist_training_x = mnist_training[:,1:].reshape((60000,28,28))
mnist_dict_train = {}
mnist_dict_test = {}
modelst = []
print("get Mnist dataset")
for num in range(10):
    mnist_dict_train[num], _ = get_x_y(mnist_training[mnist_training[:, 0] == num])
    mnist_dict_test[num], _ = get_x_y(mnist_test[mnist_test[:, 0] == num])
    modelst.append(GMM(k=5, max_iter=500))
    modelst[-1].fit(mnist_dict_train[num])
    print(num)
plt.imshow(mnist_training_x[0])
plt.show()