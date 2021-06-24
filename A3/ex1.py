import numpy as np
import matplotlib.pyplot as plt


def get_p_classes(y):
    """
    :param y: list of labels ex: [1, 0, 1 ,0 ,1]
    :return: a dict {key: probability}, key: lables  probability of each lable ex: {1: 0.6, 0: 0.4}
    """
    n_data = y.shape[0]
    p_classes = {}
    for i in y:
        if i not in p_classes:
            p_classes[i] = 1/ n_data
        else:
            p_classes[i] += 1/ n_data
    return p_classes


def misclassification_error(p_classes):
    return np.min(p_classes)


def gini_coefficient(p_classes):
    return np.cumprod(p_classes)[-1]


def entropy(p_classes):
    corss_entropy = -np.sum(p_classes[p_classes != 0] * np.log2(p_classes[p_classes != 0]))
    return corss_entropy


def cal_accuracy(y_hat, y):
    return 1 - np.sum(np.logical_xor(y_hat, y)) / y.shape[0]

class node:
    def __init__(self, right_child=None, left_child=None, loss_chr='entropy'):
        self.right_child = right_child
        self.left_child = left_child
        self.split_feature = None
        self.split_value = None
        self.depth = 0
        self.dominant_label = None
        self.loss_chr=loss_chr

class DecisionTree:
    # You will likely need to add more arguments to the constructor
    def __init__(self, max_depth=float('inf'), loss_chr='entropy', n_random_feature=0):
        self.root = node()
        self.depth = 0
        self.n_random_feature = n_random_feature
        self.max_depth = max_depth
        self.loss_chr = loss_chr
        self.n_points = None
        return

    def set_max_depth(self, max_depth):
        self.max_depth = max_depth

    def set_n_random_feature(self, n_random_feature):
        self.n_random_feature = n_random_feature

    def __get_loss_fn(self,loss):
        if loss == "entropy":
            return entropy
        elif loss == "gini_coefficient":
            return gini_coefficient
        elif loss == "misclassification_error":
            return misclassification_error

    def __random_pick_n_features(self, n):
        return np.array([np.random.randint(self.n_features) for _ in range(n)])

    def __get_split__(self, data_pair, loss_chr, random_feature_idx):
        loss_fn = self.__get_loss_fn(loss_chr)
        n_point = data_pair.shape[0]
        cur_loss = float('inf')
        split_feature = 0
        split_val = 0
        right_datapair = None
        left_datapair = None
        if random_feature_idx.shape[0]:
            feature_idx_lst = random_feature_idx
        else:
            feature_idx_lst = range(self.n_features)
        for feature_idx in feature_idx_lst:
            sorted_data_pair = data_pair[data_pair[:, feature_idx].argsort()]
            for data_idx in range(n_point-1):
                if not np.equal(sorted_data_pair[data_idx,feature_idx], sorted_data_pair[data_idx+1,feature_idx]):
                    left_p_classes = get_p_classes(sorted_data_pair[:, -1][:data_idx+1])
                    left_p = np.array(list(left_p_classes.values()))

                    right_p_classes = get_p_classes(sorted_data_pair[:, -1][data_idx+1:])
                    right_p = np.array(list(right_p_classes.values()))
                    loss_val = (data_idx + 1) * loss_fn(left_p) + \
                               (n_point - (data_idx + 1)) * loss_fn(right_p)
                    if loss_val < cur_loss:
                        cur_loss = loss_val
                        split_feature = feature_idx
                        split_val = sorted_data_pair[data_idx, feature_idx]
                        left_datapair = sorted_data_pair[:data_idx+1]
                        right_datapair = sorted_data_pair[data_idx+1:]
                        min_left_p_classes = left_p_classes
                        min_right_p_classes = right_p_classes
        left_label = max(min_left_p_classes, key=min_left_p_classes.get)
        right_label = max(min_right_p_classes, key=min_right_p_classes.get)
        #left_label
        return split_feature, split_val, right_datapair, left_datapair, right_label, left_label

    def __is_pure(self,y):
        if y.shape[0] == 1:
            return True
        else:
            base = y[0]
            for i in y[1:]:
                if base != i:
                    return False
        return True

    def __all_feature_are_same(self, x, random_feature_idx):
        if x.shape[0] <= 1:
            return True
        else:
            if random_feature_idx.shape[0]:
                base = x[0][random_feature_idx]
                sele_x = x[1:,random_feature_idx]
            else:
                base = x[0]
                sele_x = x[1:]
            for i in sele_x:
                if not np.array_equal(base,i):
                    return False
        return True

    def __recurrent_build__(self, data_pair, max_depth, root, loss_chr, random_feature_idx):
        if root.depth > self.depth:
            self.depth = root.depth
        if self.__is_pure(data_pair[:,-1]) or self.__all_feature_are_same(data_pair[:, :-1], random_feature_idx) or root.depth >= max_depth:
            return
        else:
            split_feature, split_val, right_datapair, left_datapair, right_label, left_label = \
                self.__get_split__(data_pair, loss_chr, random_feature_idx)
            root.split_feature = split_feature
            root.split_value = split_val
            if left_datapair.shape[0] >= 1:
                root.left_child = node()
                root.left_child.depth = root.depth+1
                root.left_child.dominant_label = left_label
                self.__recurrent_build__(left_datapair, max_depth, root.left_child, loss_chr, random_feature_idx)


            if right_datapair.shape[0] >= 1:
                root.right_child = node()
                root.right_child.depth = root.depth+1
                root.right_child.dominant_label = right_label
                self.__recurrent_build__(right_datapair, max_depth, root.right_child, loss_chr, random_feature_idx)

    def build(self, X, y):
        self.n_points = y.shape[0]
        data_pair = np.c_[X,y]
        self.n_features = X.shape[1]
        Node = self.root
        root_p_classes=get_p_classes(y)
        root_label = max(root_p_classes, key=root_p_classes.get)
        Node.dominant_label = root_label
        if self.n_random_feature == "all" or self.n_features <= self.n_random_feature:
            random_feature_idx = np.array([])
        else:
            random_feature_idx = self.__random_pick_n_features(self.n_random_feature)
        self.__recurrent_build__(data_pair, self.max_depth, Node, self.loss_chr, random_feature_idx)
        return

    def predict_x(self, root, x, max_depth):
        feature = root.split_feature
        value = root.split_value
        if (feature == None or value == None or root.depth == max_depth):
            return root.dominant_label
        try:
            if x[feature] > value:
                if root.right_child:
                    return self.predict_x(root.right_child, x, max_depth)
                else:
                    return root.dominant_label
            else:
                if root.left_child:
                    return self.predict_x(root.left_child, x, max_depth)
                else:
                    return root.dominant_label
        except:
            print(feature)

    def predict(self, X, max_depth=float('inf')):
            result = np.array([])
            for x in X:
                y_hat = self.predict_x(self.root, x, max_depth)
                result = np.append(result, y_hat)
            return result


X_train = np.loadtxt('data/X_train.csv', delimiter=",")
y_train = np.loadtxt('data/y_train.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test.csv', delimiter=",")
y_test = np.loadtxt('data/y_test.csv', delimiter=",").astype(int)

loss_fn_list = ["entropy", "gini_coefficient", "misclassification_error"]
training_acc_dict = {}
test_acc_dict = {}
for loss_fn in loss_fn_list:
    DT = DecisionTree(max_depth=float('inf'), loss_chr=loss_fn, n_random_feature='all')
    # n_random_feature: 'all', select all features, #, number of random features would use.
    DT.build(X=X_train, y=y_train)
    training_acc = []
    test_acc=[]
    for depth in range(DT.depth+1):
        Y_train_hat = DT.predict(X=X_train, max_depth=depth)
        training_accuracy = cal_accuracy(Y_train_hat, y_train)
        training_acc.append(training_accuracy)
        Y_train_hat = DT.predict(X=X_test, max_depth=depth)
        test_accuracy = cal_accuracy(Y_train_hat, y_test)
        test_acc.append(test_accuracy)
    training_acc_dict[loss_fn] = training_acc
    test_acc_dict[loss_fn] = test_acc
for curve_name in training_acc_dict:
    plt.title(curve_name)
    plt.xlabel("depth of the tree")
    plt.ylabel("accuracy")
    plt.plot(training_acc_dict[curve_name])
    plt.plot(test_acc_dict[curve_name])
    plt.grid(True, fillstyle='full')
    plt.show()


class RandomForest:
    def __init__(self, n_trees, loss_chr, max_depth=float('inf'), n_random_feature=0):
        self.n_trees = n_trees
        self.forest = [DecisionTree(max_depth=max_depth, loss_chr=loss_chr, n_random_feature=n_random_feature) for _ in range(n_trees)]

    def __random_pick_n_points(self, n_points, n):
        return np.array([np.random.randint(n_points) for _ in range(n)])

    def build(self, X, y):
        for i in range(self.n_trees):
            select_idx = self.__random_pick_n_points(y.shape[0], y.shape[0])
            self.forest[i].build(X=X[select_idx], y=y[select_idx])

    def predict_x(self, X, max_depth=float('inf')):
        result=np.array([])
        for i in range(self.n_trees):
            result = np.append(result, self.forest[i].predict_x(self.forest[i].root, X, max_depth=max_depth))
        n_classes = get_p_classes(result)
        return max(n_classes, key=n_classes.get)

    def predict(self, X):
        result=np.array([])
        for i in range(X.shape[0]):
            result = np.append(result, self.predict_x(X[i], max_depth = float('inf')))
        return result


class BaggingTree:
    def __init__(self, n_trees, loss_chr, max_depth=float('inf')):
        self.n_trees = n_trees
        self.forest = [DecisionTree(max_depth=max_depth, loss_chr=loss_chr, n_random_feature=0) for _ in range(n_trees)]

    def __random_pick_n_points(self, n_points, n):
        return np.array([np.random.randint(n_points) for _ in range(n)])

    def build(self, X, y):
        for i in range(self.n_trees):
            select_idx = self.__random_pick_n_points(y.shape[0], y.shape[0])
            self.forest[i].build(X=X[select_idx], y=y[select_idx])

    def predict_x(self, X, max_depth=float('inf')):
        result=np.array([])
        for i in range(self.n_trees):
            result = np.append(result, self.forest[i].predict_x(self.forest[i].root, X, max_depth=max_depth))
        n_classes = get_p_classes(result)
        return max(n_classes, key=n_classes.get)

    def predict(self, X):
        result=np.array([])
        for i in range(X.shape[0]):
            result = np.append(result, self.predict_x(X[i], max_depth = float('inf')))
        return result


print("#-------------------------DecisionTree-------------------------#")
training_acc_lst = np.array([])
test_acc_lst = np.array([])
for _ in range(11):
    BT = BaggingTree(n_trees=101, loss_chr="entropy", max_depth=3)
    BT.build(X=X_train, y=y_train)
    y_hat = BT.predict(X_train)
    training_acc = cal_accuracy(y_hat, y_train)
    training_acc_lst = np.append(training_acc_lst, training_acc)
    y_hat = BT.predict(X_test)
    test_acc = cal_accuracy(y_hat, y_test)
    test_acc_lst = np.append(test_acc_lst, test_acc)
    print("training_acc: ", training_acc, "test_acc: ", test_acc)
print("#-------------------------DecisionTree-------------------------#")
print("minimum test accuracy", np.min(test_acc_lst))
print("median test accuracy", np.median(test_acc_lst))
print("maximum test accuracy", np.max(test_acc_lst),"\n\n")

print("#-------------------------RandomForest-------------------------#")
training_acc_lst = np.array([])
test_acc_lst = np.array([])
for _ in range(11):
    RF = RandomForest(n_trees=101, loss_chr="entropy", n_random_feature=4, max_depth=3)
    RF.build(X=X_train, y=y_train)
    y_hat = RF.predict(X_train)
    training_acc_lst = training_acc = cal_accuracy(y_hat, y_train)
    np.append(training_acc_lst, training_acc)
    y_hat = RF.predict(X_test)
    test_acc = cal_accuracy(y_hat, y_test)
    test_acc_lst = np.append(test_acc_lst, test_acc)
    print("training_acc: ", training_acc, "test_acc: ", test_acc)
print("#-------------------------RandomForest-------------------------#")
print("minimum test accuracy", np.min(test_acc_lst))
print("median test accuracy", np.median(test_acc_lst))
print("maximum test accuracy", np.max(test_acc_lst))

