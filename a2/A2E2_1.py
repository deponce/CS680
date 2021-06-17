import numpy as np
import statsmodels.api as sm
from sklearn import svm

X_train_A = np.genfromtxt('data/X_train_A.csv', delimiter=',')

Y_train_A = np.genfromtxt('data/Y_train_A.csv', delimiter=',')

clf = svm.SVC(kernel='linear', C=1)

clf.fit(X_train_A, Y_train_A)
print("soft-margin SVM weight vector: ", clf.coef_, "\nsoft-margin SVM bias: ", clf.intercept_)

print("n_sv",clf.n_support_)

clf = svm.SVC(kernel='linear', C=float('inf'))

clf.fit(X_train_A, Y_train_A)
print("hard-margin SVM weight vector: ", clf.coef_, "\nhard-margin SVM bias: ", clf.intercept_)

print("n_sv",clf.n_support_)

sm.Logit(Y_train_A, X_train_A).fit()


