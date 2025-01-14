import numpy as np

#Exercise 2
#Usage: python3 ex2.py
#Load the files in some other way (e.g., pandas) if you prefer
X_train_A = np.loadtxt('data/X_train_A.csv', delimiter=",")
Y_train_A = np.loadtxt('data/Y_train_A.csv',  delimiter=",").astype(int)

X_train_B = np.loadtxt('data/X_train_B.csv', delimiter=",")
Y_train_B = np.loadtxt('data/Y_train_B.csv', delimiter=",").astype(int)
X_test_B = np.loadtxt('data/X_test_B.csv', delimiter=",")
Y_test_B = np.loadtxt('data/Y_test_B.csv', delimiter=",").astype(int)

# E2.1 run logistic regression, soft/ hard margin SVM on mystery dataset A
print("#----------------------E2.1---------------------------------#")
import statsmodels.api as sm    # using Logit in statsmodels
import statsmodels
from sklearn import svm     # using svm from sklearn

print("### start to run soft-margin SVM ###")
try:
    soft_margin_svm = svm.SVC(kernel='linear', C=1)
    soft_margin_svm.fit(X_train_A, Y_train_A)
except Exception as pse:
    print(pse)
print("soft-margin SVM weight vector: ", soft_margin_svm.coef_, "\nsoft-margin SVM bias: ",
      soft_margin_svm.intercept_)
print("num of support vectors",soft_margin_svm.n_support_)

print("\n### start to run hard-margin SVM ###")
try:
    hard_margin_svm = svm.SVC(kernel='linear', C=float('inf'))
    hard_margin_svm.fit(X_train_A, Y_train_A)
except Exception as pse:
    print(pse)
print("hard-margin SVM weight vector: ", hard_margin_svm.coef_, "\nhard-margin SVM bias: ",
      hard_margin_svm.intercept_)
print("num of support vectors",hard_margin_svm.n_support_)

print("\n### start to run logistic regression ###")
try:
    sm.Logit(Y_train_A, X_train_A).fit()
except Exception as pse:
    print(pse)

print("\n#----------------------E2.2---------------------------------#")

def rescale_to_np_1(data):
    """
    :param data: input, numpy array
    :return: scaled numpy array [-1, 1]
    only for dataset $\y_i \in\{0, 1\}$
    """
    maximum = data.max()
    minimun = data.min()
    return 2*(data-minimun)/(maximum-minimun)-1

def less_1(arr, thr = 1e-10):
    """
    :param arr: input, 1-D numpy array
    :param thr: the thr of the num resolution
    :return: num of entry less than 1 in array
    """
    cnt = 0
    for e in arr:
        if e <= 1+thr:
            cnt = cnt + 1
    return cnt


coefficient = soft_margin_svm.coef_
y_hat_A = X_train_A.dot(coefficient[0])  # y_hat_A is the prediction on the test set
Y_train_A_scaled = rescale_to_np_1(Y_train_A)  # replace 0's with -1's
p_error = Y_train_A_scaled * y_hat_A
# p_error: prediction error, in soft-margin SVM
# 1-y\hat{y}<= 0: on the right side of the margin
# 0 <= 1-y\hat{y} <=1 : with in the margin
# y\hat{y < 0 : incorrect

nerror = less_1(p_error,1e-10)  # count the number less than 1

print("number of values <=1: ", nerror)  # how many of these values are <=1?
print("num of support vectors: ", np.sum(soft_margin_svm.n_support_), soft_margin_svm.n_support_)
print("alpha*label : ", soft_margin_svm.dual_coef_)
print("support vector: ", X_train_A[soft_margin_svm.support_])
print("parameter weight vector:", soft_margin_svm.coef_)


# compute three method on dataset B

print("\n### start to run soft-margin SVM on dataset B###")
soft_margin_svm = svm.SVC(kernel='linear', C=1)
soft_margin_svm.fit(X_train_B, Y_train_B)
sm_SVM_Y_hat_B = soft_margin_svm.predict(X_test_B)
Y_error = rescale_to_np_1(sm_SVM_Y_hat_B)* rescale_to_np_1(Y_test_B)
n_Y_error = Y_error[Y_error > 0].shape[0]
print("soft margin svm empirical prediction accuracy: ", n_Y_error/Y_error.shape[0])
print("Done.")
"""
print("\n### start to run hard-margin SVM on dataset B###")
hard_margin_svm = svm.SVC(kernel='linear', C=float('inf'))
hard_margin_svm.fit(X_train_B, Y_train_B)
print("Done.")
"""
print("\n### start to run logistic regression on dataset B###")
logistic_regression = sm.Logit(Y_train_B, X_train_B).fit()
Y_hat_B = logistic_regression.predict(X_test_B)
Y_error = rescale_to_np_1(Y_hat_B) * rescale_to_np_1(Y_test_B)
n_Y_error = Y_error[Y_error > 0].shape[0]
print("logistic regression empirical prediction accuracy: ", n_Y_error/Y_hat_B.shape[0])
print("Done.")

print("num of support soft-margin SVM: ",np.sum(soft_margin_svm.n_support_), soft_margin_svm.n_support_,'\n')

print("#----------------------E2.3---------------------------------#\n")
print("#    start generate dataset    #")
def generate_ds(n_data=100, n_feature=1, scale=100):
    n_pos_data = n_data//2
    n_neg_data = n_data - n_pos_data
    weight = np.random.rand(n_feature).reshape(n_feature,1)
    bias = np.random.rand(1)
    datas = np.array([])
    lables = np.array([])
    for i in range(n_pos_data):
        n_pos_data_seeds = scale*(2*np.random.rand(n_feature-1).reshape(n_feature-1,1)-1)
        last_dim=(1-bias- weight[:-1].T.dot(n_pos_data_seeds))/weight[-1]
        data = np.append(n_pos_data_seeds, last_dim)
        datas=np.append(datas, data)
        lables=np.append(lables, 1)
    for i in range(n_neg_data):
        n_pos_data_seeds = scale * (2 * np.random.rand(n_feature-1).reshape(n_feature-1,1) - 1)
        last_dim=(-1-bias- weight[:-1].T.dot(n_pos_data_seeds))/weight[-1]
        data = np.append(n_pos_data_seeds, last_dim)
        datas=np.append(datas, data)
        lables=np.append(lables, -1)
    return datas.reshape(n_data,n_feature), lables

while True:
    X_train_A,Y_train_A = generate_ds(132,20,10)
    clf = svm.SVC(kernel='linear', C=float('inf'))
    clf.fit(X_train_A, Y_train_A)
    num_point = np.sum(clf.n_support_)
    if num_point > 100:
        pot_x = X_train_A[clf.support_]
        pot_y = Y_train_A[clf.support_]
        clf.fit(pot_x, pot_y)
        if(np.sum(clf.n_support_) == num_point):
            break

np.savetxt("data/generate_data_X.cvs",pot_x, delimiter=",")
np.savetxt("data/generate_data_Y.cvs",pot_y, delimiter=",")

X_train = np.loadtxt('data/generate_data_X.cvs', delimiter=",")
Y_train = np.loadtxt('data/generate_data_Y.cvs', delimiter=",")
clf = svm.SVC(kernel='linear', C=float('inf'))
clf.fit(X_train_A, Y_train_A)
print("There are ", Y_train.shape[0], "opints in the generated dataset.")
print(np.sum(clf.n_support_), " of them are support vectors")
print("#    Done!    #")
