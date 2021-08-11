from sklearn.mixture import GaussianMixture
import numpy as np

mnist_training = np.loadtxt("./data/mnist_train.csv", delimiter=',')
mnist_test = np.loadtxt("./data/mnist_test.csv", delimiter=',')
def get_x_y(mnist):
    y = mnist[:,0]
    x = mnist[:,1:]
    return x, y
mnist_training_0, Y = get_x_y(mnist_training[mnist_training[:,0]==0])
mnist_test_0, test_Y = get_x_y(mnist_test[mnist_test[:,0]==0])
gm = GaussianMixture(n_components=5, random_state=0, covariance_type='diag').fit(mnist_training_0)
gm.predict(mnist_test_0[0])
