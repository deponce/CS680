from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import statsmodels.api as sm
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.svm import SVC


X_train_A = pd.read_csv('data/X_train_A.csv',header=None)
Y_train_A = pd.read_csv('data/Y_train_A.csv',  header=None)
X_train_A1 = X_train_A[:]
X_train_B = pd.read_csv('data/X_train_B.csv',header=None)
Y_train_B = pd.read_csv('data/Y_train_B.csv',  header=None)
X_test_B = pd.read_csv('data/X_test_B.csv',header=None)
Y_test_B = pd.read_csv('data/Y_test_B.csv',  header=None)


print("-----Exercise 2.2-datasetA-----")
####Exercise 2.2-datasetA####

# svm
svm_clf = SVC(kernel='linear', C=1)
res = svm_clf.fit(X_train_A, Y_train_A)
res_pre = svm_clf.predict(X_train_A[:])
co_vector = svm_clf.coef_[0]#w

# coefficient vector
res_coef = X_train_A.dot(co_vector.T)
key_feature = svm_clf.support_vectors_
print(key_feature)
print(res_coef)

# help emphasize a few key feature
keyfeatuer = PCA(n_components = None)
X_train_pca = keyfeatuer.fit_transform(X_train_A1)
print(keyfeatuer.explained_variance_ratio_)

# detect 0 and replace
u = []
for i in range(len(res_coef)):
    if res_coef[i] == 0:
        u.append(res_coef[i])
print(len(u))
print(0 in res_coef)

# How many of these values are ≤ 1?
m = 0
for i in range(len(res_coef)):
    if res_coef[i] <= 1:
        m = m + 1
    else:
        m = m
print("how many values are <=1?", m)

# sign of each point’s label
i = 0
for i in range(len(res_coef)):
    if res_coef[i] <= 0:
        res_coef[i] = -1
    else:
        res_coef[i] = 1

# plot
plt.scatter(X_train_A[1], res_coef, label='0')
plt.show()


