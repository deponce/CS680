import numpy as np
from model.models import *
import gzip
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
gmm_dataset = np.loadtxt("./data/gmm_dataset.csv", delimiter=",")
loss_1_10=[]
k_s = range(1,11)
modelst = []
for k in tqdm(k_s):
    modelst.append(GMM(k=k, max_iter=500, tol=1e-5))
    loss = modelst[-1].fit(gmm_dataset)
    loss_1_10.append(loss[-1])
    #plt.plot(loss)
    #plt.show()
plt.plot(k_s, loss_1_10)
plt.xlabel("k")
plt.ylabel("loss")
plt.show()

print("I would choose k = 5")
Pi = modelst[4].Pi
Mu = modelst[4].Mu
S = modelst[4].S
i = np.argsort(Pi)
print("mixing weights: ", Pi[i])
np.savetxt('Pi.csv',Pi[i].astype('float16').T,delimiter=',')
print("mean vector: ", Mu[i, :])
np.savetxt('Mu.csv',Mu[i].astype('float16').T,delimiter=',')
print("variance vector", S[i, :])
np.savetxt('S.csv',S[i].astype('float16').T,delimiter=',')
