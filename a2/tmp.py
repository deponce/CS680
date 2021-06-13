import numpy as np
from sklearn import svm
from sklearn import decomposition
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

X_train_A = np.genfromtxt('data/X_train_A.csv', delimiter=',')

Y_train_A = np.genfromtxt('data/Y_train_A.csv', delimiter=',')

clf = svm.SVC(kernel='linear', C=1)

clf.fit(X_train_A, Y_train_A)

pca = decomposition.PCA(n_components=3)
xs = pca.fit(X_train_A).transform(X_train_A)

"""
fig = plt.figure()
ax1 = plt.axes(projection='3d')



ax1.scatter3D(xs.T[0],xs.T[1],xs.T[2])
plt.show()
"""
plt.scatter(xs.T[0],xs.T[1])
plt.show()
# A2E2_2

coefficient = clf.coef_

y_hat_A = X_train_A.dot(coefficient[0])+clf.intercept_
