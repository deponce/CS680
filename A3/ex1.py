import numpy as np
import copy

def get_p_classes(y):
    n_data = y.shape[0]
    classes = {}
    for i in y:
        if i not in classes:
            classes[i] = 0
        else:
            classes[i] += 1
    p_classes = np.array(list(classes.values())) / n_data
    return p_classes

def misclassification_error(p_classes):
    return np.min(p_classes)

def gini_coefficient(p_classes):
    return 1 - p_classes.dot(p_classes)

def entropy(p_classes):
    corss_entropy = -np.sum(p_classes[p_classes != 0] * np.log2(p_classes[p_classes != 0]))
    return corss_entropy

class node:
    def __init__(self, right_child=None, left_child=None):
        self.right_child = right_child
        self.left_child = left_child
        self.split_feature = 0
        self.split_value = 0
        self.depth = 0
        self.dominant_label = None

class DecisionTree:
    # You will likely need to add more arguments to the constructor
    def __init__(self):
        self.root = node()
        self.depth = 0
        return
    def __get_loss_fn(self,loss):
        if loss == "entropy":
            return entropy
        elif loss == "gini_coefficient":
            return gini_coefficient
        elif loss == "misclassification_error":
            return misclassification_error

    def __get_split__(self, data_pair, loss_chr):
        loss_fn = self.__get_loss_fn(loss_chr)
        n_point = data_pair.shape[0]
        cur_loss = float('inf')
        split_feature = 0
        split_val = 0
        right_datapair = None
        left_datapair = None
        rt_right_p = None
        rt_left_p = None
        for feature_idx in range(self.n_features):
            sorted_data_pair = data_pair[data_pair[:,feature_idx].argsort()]
            for data_idx in range(1, n_point):
                left_p = get_p_classes(sorted_data_pair[:,-1][:data_idx])
                right_p = get_p_classes(sorted_data_pair[:,-1][data_idx:])
                loss_val = data_idx*loss_fn(left_p)+\
                           (n_point-data_idx)*loss_fn(right_p)
                if loss_val < cur_loss:
                    cur_loss = loss_val
                    split_feature = feature_idx
                    split_val = data_pair[data_idx, feature_idx]
                    right_datapair = sorted_data_pair[:data_idx]
                    left_datapair = sorted_data_pair[data_idx:]
                    rt_left_p = left_p
                    rt_right_p = right_p
        #left_label
        return split_feature, split_val, right_datapair, left_datapair, rt_right_p, rt_left_p

    def __is_pure(self,y):
        if y.shape[0] == 1:
            return True
        else:
            base = y[0]
            for i in y[1:]:
                if base != i:
                    return False
        return True
    def __all_feature_are_same(self,x):
        if x.shape[0] <= 1:
            return True
        else:
            base = x[0]
            for i in x[1:]:
                if not np.array_equal(base,i):
                    return False
        return True
    def __recurrent_bulid__(self, data_pair, root, loss_chr):
        if self.__is_pure(data_pair[:,-1]) or self.__all_feature_are_same(data_pair[:,:-1]):
            return
        else:
            split_feature, split_val, right_datapair, left_datapair, right_p, left_p = \
                self.__get_split__(data_pair, loss_chr)
            root.split_feature = split_feature
            root.split_value = split_val
            if left_datapair.shape[0] >= 1:
                root.left_child = node()
                root.left_child.depth = root.depth+1
                self.__recurrent_bulid__(left_datapair, root.left_child, loss_chr)

            if right_datapair.shape[0] >= 1:
                root.right_child = node()
                root.right_child.depth = root.depth+1
                self.__recurrent_bulid__(right_datapair, root.right_child, loss_chr)

    def build(self, X, y, loss_chr):
        data_pair = np.c_[X,y]
        self.n_features = X.shape[1]
        Node = self.root
        self.__recurrent_bulid__(data_pair, Node, loss_chr)
        return

    def predict(self, X):
        # Implement me!
        return

    # Load data


X_train = np.loadtxt('data/X_train.csv', delimiter=",")
y_train = np.loadtxt('data/y_train.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test.csv', delimiter=",")
y_test = np.loadtxt('data/y_test.csv', delimiter=",").astype(int)


DT = DecisionTree()
DT.build(X_train, y_train, "entropy")
print(DT)
