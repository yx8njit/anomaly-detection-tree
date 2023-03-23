import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

'''
An ICAD-Tree in Newick format (which requires all nodes to be named)
To use the tree:
Step 1: construct a tree from a Newick tree string
Step 2: train the model by passing in training data and training labels
Step 3: run inference by passing in test data and test labels
Example:
>>tr = ICADTree(' (Neuron2, Neuron3,(Neuron1-1,Neuron1-2)Neuron1)Neurons ')
>>tr.train(x_train, y_train)
>>test_scores, test_pvals = tr.inference(x_testall, y_testall)
'''
class ICADTree:
    def __init__(self, root_str_raw):
        root_str = root_str_raw.replace(" ", "")
        if '(' not in root_str:
            self.is_leaf = True
            self.name = root_str
            self.children = []
        else:
            self.is_leaf = False
            self.name = root_str.split(')')[-1]
            children_str = root_str[1:-(len(self.name)+1)]
            self.children = self.__resolve_children_from_str(children_str)
        self.train_data, self.val_data, self.train_labels, self.val_labels = [], [], [], []
        self.model = None
        self.cal_scores = []

    # an internal method that resolves childresn from a string in Newick format
    # the returned values are a list of tree objects, each corresponds to a child of the current node
    def __resolve_children_from_str(self, children_str):
        children = []
        index = 0
        cur_child_str = ""
        left_parentheses_count = 0
        while index < len(children_str):
            if children_str[index] == ',' and left_parentheses_count ==0:
                children.append(ICADTree(cur_child_str))
                cur_child_str = ""
            else:
                if children_str[index] == '(':
                    left_parentheses_count += 1
                elif children_str[index] == ')':
                    left_parentheses_count -= 1
                cur_child_str += children_str[index]
            index += 1
        children.append(ICADTree(cur_child_str))
        return children

    # Get the list of descendants of the current node
    # The descendants are also in Newick tree object
    # No particular ordering of the descendants is assumed.
    def get_descendants(self):
        descendants = []
        if self.children is not None:
            descendants.extend(self.children)
            for c in self.children:
                c_descendants = c.get_descendants()
                if len(c_descendants)>0:
                    descendants.extend(c_descendants)
        return descendants

    # An internal method that attaches training data to the tree
    # By data, it includes both training data and validation data
    # It is supposed to be called by the training method
    def __attach_data(self, data, labels):
        all_names = set([d.name for d in self.get_descendants()])
        all_names.add(self.name)
        indices = [i for i,e in enumerate(labels) if e in all_names]
        cur_data = data[indices]
        cur_labels = labels[indices]
        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(cur_data, cur_labels,
                                                                                              test_size=0.3, random_state=52)
        for c in self.children:
            c.__attach_data(cur_data, cur_labels)

    def print_leaf(self):
        for child in self.children:
            if child.is_leaf:
                print(child.name, end=",")
            else:
                child.print_leaf()

    def __str__(self) -> str:
        if self != None and self.children != None:
            return self.name + "->" + "("+ ",".join([str(c) for c in self.children])+ ")"

    def train(self, data, labels):
        self.__attach_data(data, labels)
        self.model = OneClassSVM(gamma='scale', kernel='rbf').fit(self.train_data)
        self.cal_scores = -self.model.decision_function(self.val_data)

    def inference(self, test_data, test_labels):
        test_scores = -self.model.decision_function(test_data)
        cur_pvals = []
        for i in range(len(test_labels)):
            cur_pvals.append((np.sum(self.cal_scores >= test_scores[i]) + 1) / (len(self.cal_scores) + 1))
        return test_scores, cur_pvals


if __name__ == "__main__":
    # read a csv file with label
    data = np.loadtxt('../mousebrain/mousebrain_simulate3_pcalarge.txt')
    #data = data.T
    labels = np.loadtxt('../mousebrain/mousebrain_simlabel3_large.txt', dtype = 'str')

    # divided into outlier and inlier
    name = ['Neuron3','Neuron1-2','Neuron1-1','Neuron2']
    index = [i for i,e in enumerate(labels) if e in name]
    x_inlier = data[index]
    y_inlier = [1] * x_inlier.shape[0]
    y_labels = labels[index]
    #d = dict([(y,x+1) for x,y in enumerate(sorted(set(y_labels)))])
    #y_labelsum = np.array([d(x) for x in y_labels])

    # outlier
    name = ['Microglia','Oligo2','Oligo1','Astrocytes']
    index = [i for i,e in enumerate(labels) if e in name]
    x_outlier = data[index]
    y_outlier = [2]*x_outlier.shape[0]
    num_list = random.sample(range(0, 1600), 100)
    #print(num_list)
    x_outlier = x_outlier[num_list]
    y_outlier = [2]*x_outlier.shape[0]
    #print(x_outlier.shape)

    # seperate inlier to train and test data
    x_train, x_test, y_train, y_test = train_test_split(x_inlier, y_labels,
                                                        test_size = 0.5, random_state=52)
    #x_trainset, x_calibration, y_trainset, y_calibration = train_test_split(x_train, y_train,
    #test_size=0.3, random_state=52)
    print(x_train.shape, x_test.shape)
    x_testall = np.concatenate([x_test, x_outlier], axis = 0)
    y_testall = np.concatenate([y_test, y_outlier], axis = 0)

    tr = ICADTree(' (Neuron2, Neuron3,(Neuron1-1,Neuron1-2)Neuron1)Neurons ')
    # for c in tr.get_descendants():
    #     print(c.name)
    # print(tr)
    tr.train(x_train, y_train)
    #print(tr)
    a, b = tr.inference(x_testall, y_testall)
    print(a)
    print(b)
    # print(len(tr.cal_scores))
    # tr.print_leaf()
    # for c in tr.children:
    #     print(c)