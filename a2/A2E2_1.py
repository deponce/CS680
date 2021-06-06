import numpy as np

import statsmodels.api as sm

from matplotlib import pyplot as plt
from sklearn import svm

X_train_A = np.genfromtxt('data/X_train_A.csv', delimiter=',')

Y_train_A = np.genfromtxt('data/Y_train_A.csv', delimiter=',')

clf = svm.SVC(kernel='linear', C=1)


clf.fit(X_train_A, Y_train_A)
print(np.sum(clf.predict(X_train_A)-Y_train_A))

sm.Logit(Y_train_A, X_train_A).fit()



