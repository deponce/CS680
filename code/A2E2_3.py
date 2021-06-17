import numpy as np

from sklearn import svm

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

np.savetxt("data/generate_data_X.cvs",X_train_A[clf.support_], delimiter=",")
np.savetxt("data/generate_data_Y.cvs",Y_train_A[clf.support_], delimiter=",")

X_train = np.loadtxt('data/generate_data_X.cvs', delimiter=",")
Y_train = np.loadtxt('data/generate_data_Y.cvs', delimiter=",")

clf = svm.SVC(kernel='linear', C=float('inf'))
clf.fit(X_train, Y_train)
print("num of training data:", Y_train.shape[0])
print("num of support vector:", np.sum(clf.n_support_))


