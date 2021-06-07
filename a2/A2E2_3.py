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

X_train_A,Y_train_A = generate_ds(132,20,10)

clf = svm.SVC(kernel='linear', C=float('inf'))

clf.fit(X_train_A, Y_train_A)
print(clf.n_support_)