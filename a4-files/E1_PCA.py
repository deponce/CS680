import numpy as np
from model.models import *
import gzip
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.special import logsumexp
import numpy as np
from sklearn.decomposition import PCA
dim_keep = 30
print("load Mnist dataset")
start = time.time()
####
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform to normalized Tensors
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

mnist_training_x = next(iter(train_loader))[0].numpy()
mnist_test_x = next(iter(test_loader))[0].numpy()

mnist_training_x = 255*mnist_training_x.reshape([mnist_training_x.shape[0],mnist_training_x.shape[2]*mnist_training_x.shape[3]])
mnist_test_x = 255*mnist_test_x.reshape([mnist_test_x.shape[0],mnist_test_x.shape[2]*mnist_test_x.shape[3]])
print("applying PCA")

pca = PCA()

mnist_training_x = pca.fit_transform(mnist_training_x)[:, :dim_keep]

mnist_test_x = pca.transform(mnist_test_x)[:, :dim_keep]


mnist_training_y = next(iter(train_loader))[1].numpy()
mnist_test_y = next(iter(test_loader))[1].numpy()
####
#mnist_training = np.loadtxt("./data/mnist_train.csv", delimiter=',')
#mnist_test = np.loadtxt("./data/mnist_test.csv", delimiter=',')
mnist_dict_train = {}
mnist_dict_test = {}
modelst = []

for num in range(10):
    mnist_dict_train[num] = mnist_training_x[mnist_training_y == num]
    mnist_dict_test[num] = mnist_test_x[mnist_test_y == num]
n_train_data = mnist_training_x.shape[0]
n_test_data = mnist_test_x.shape[0]
print("get Mnist dataset")
end = time.time()
print(end-start,"s")

accs = []
krange = range(1, 21)

for k in krange:
    modelst = []
    start = time.time()
    for num in tqdm(range(10)):
        S_scale = np.max(mnist_dict_train[num], axis=0)**2
        modelst.append(GMM(k=k, max_iter=500, name=num, tol=1e-5, S_scale=S_scale, using_S_thr=True, S_thr=1/n_train_data*pca.explained_variance_ratio_[dim_keep]))
        loss = modelst[-1].fit(mnist_dict_train[num])
        #plt.plot(loss)
        #plt.show()

    C_prob = []
    log_probs = []
    for num in range(10):
        C_prob.append(mnist_dict_train[num].shape[0]/n_test_data)
    C_prob = np.array(C_prob)
    for model_idx, model in enumerate(modelst):
        log_prob = model.predict(mnist_test_x)
        log_probs.append(log_prob)
    log_probs = np.array(log_probs)
    log_probs = log_probs - np.min(np.max(log_probs, axis=0), axis=0)

    probs = (logsumexp(log_probs.T+np.log(C_prob), axis=1)).T
    maxim = np.argmax(probs, axis=0)
    acc = len(maxim[maxim == mnist_test_y]) / len(maxim)
    accs.append(acc)
    print(k, "acc is: ", acc, "\n")

accs = np.array(accs)
plt.plot(krange,1-accs)
plt.xlabel("k")
plt.ylabel("loss rate")
plt.show()
