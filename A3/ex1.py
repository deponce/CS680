import numpy as np

class node:
    def __init__(self, parent=None, right_child=None, left_child=None,labels=np.array([]), data_x=np.array([])):
        self.parent = parent
        self.right_child = right_child
        self.left_child = left_child
        self.labels = labels
        self.data_x = data_x
        self.__n_of_each_labels= {}  # the number of each labels
        self.__n_data = labels.shape[0]  # the number of input data points
        self.__n_lables = self.labels.shape[0]
        self.depth = 0
        for i in self.labels:
            if i not in self.__n_of_each_labels:
                self.__n_of_each_labels[i] = 0
            else:
                self.__n_of_each_labels[i] += 1
    def set_depth(self, depth):
        self.depth = depth
    def right_child_is(self, right_child=None):
        self.right_child = right_child
        right_child.parent = self.node
        right_child.set_depth(self.depth+1)
    def left_child_is(self, left_child=None):
        self.left_child = left_child
        left_child.parent = self.node
        left_child.set_depth(self.depth + 1)
    def get_data(self):
        return self.data_x
    def get_labels(self):
        return self.labels
    def misclassification_error(self):
        n_classes = np.array(list(self.__n_of_each_labels))
        p_classes = n_classes / self.__n_lables
        return np.min(p_classes)
    def gini_coeddicient(self):
        n_classes = np.array(list(self.__n_of_each_labels))
        p_classes = n_classes/self.__n_lables
        return 1 - p_classes.dot(p_classes)

    def entropy(self):

        return 0



class DecisionTree:
    #You will likely need to add more arguments to the constructor
    def __init__(self):
        self.root = node(None, None, )
        #Implement me!
        return

    def build(self, X, y):
        #Implement me!
        return
    
    def predict(self, X):
        #Implement me!
        return 

#Load data
X_train = np.loadtxt('data/X_train.csv', delimiter=",")
y_train = np.loadtxt('data/y_train.csv', delimiter=",").astype(int)
X_test = np.loadtxt('data/X_test.csv', delimiter=",")
y_test = np.loadtxt('data/y_test.csv', delimiter=",").astype(int)

print(y_test)

