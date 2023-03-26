from sklearn import svm
import random

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

'''
An BCOPS-Tree in Newick format (which requires all nodes to be named)
To use the tree:
Step 1: construct a tree from a Newick tree string 
Step 2: attach the training data and training labels
Step 3: run inference by passing in the test data
Example: 
>>tr = BCOPSTree(' (Neuron2, Neuron3,(Neuron1-1,Neuron1-2)Neuron1)Neurons ') 
>>tr.attach_data(x_train, y_train)
>>results = tr.run_inference(x_testall)
'''
class BCOPSTree:
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
        self.train_data, self.train_labels = [], []

    # an internal method that resolves childresn from a string in Newick format
    # the returned values are a list of tree objects, each corresponds to a child of the current node
    def __resolve_children_from_str(self, children_str):
        children = []
        index = 0
        cur_child_str = ""
        left_parentheses_count = 0
        while index < len(children_str):
            if children_str[index] == ',' and left_parentheses_count ==0:
                children.append(BCOPSTree(cur_child_str))
                cur_child_str = ""
            else:
                if children_str[index] == '(':
                    left_parentheses_count += 1
                elif children_str[index] == ')':
                    left_parentheses_count -= 1
                cur_child_str += children_str[index]
            index += 1
        children.append(BCOPSTree(cur_child_str))
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

    def __str__(self) -> str:
        if self != None and self.children != None:
            #             return self.name + "[" + str(len(self.train_labels)) + "]" + "->" + "("+ ",".join([str(c) for c in self.children])+ ")"
            return self.name + "->" + "("+ ",".join([str(c) for c in self.children])+ ")"

    # Attach the training data to the tree
    def attach_data(self, data, labels):
        all_names = set([d.name for d in self.get_descendants()])
        all_names.add(self.name)
        indices = [i for i,e in enumerate(labels) if e in all_names]
        self.train_data = data[indices]
        self.train_labels = labels[indices]

        for c in self.children:
            c.attach_data(self.train_data, self.train_labels)

    # Run inference for the specified test data, by following these steps:
    # Step 1: split train data and test data, each into two subsets;
    # Step 2: train two classifiers using train1+test1 and train2+test2 respectively;
    # Step 3: run inference on two tests subsets;
    # Step 4: calc p-values on test data;
    # Step 5: divide the test set into accepted and rejected subsets;
    # Step 6: for those accepted, check if stop condition is met;
    #     6a: if to continue, pass to the next level;
    #     6b: if to stop, assign the current label and return;
    # Step 7: for those rejected, assign outlier label and return;
    # Return values: a dict for all test data, where key is index, value is the set of labels;
    # Once a parent(i.e., the current node) get the return values of its children, it does some label consolidation.
    # Specifically:
    # if a data is not accepted by the parent, the label is outlier;
    # if it is accepted by the parent and at least one descendant, the label(s) are assigned by that descendant;
    # if it is accepted by the parent but none of its children, the label is outlier;
    def run_inference(self, test_data, alpha = 0.2/3):
        results = [None] * test_data.shape[0]

        train_data1, train_data2 = train_test_split(self.train_data, test_size = 0.5, random_state=52)
        test_data1, test_data2 = train_test_split(test_data, test_size = 0.5, random_state=52)

        model_data1 = np.concatenate((train_data1, test_data1), axis = 0)
        model_label1 = ['a']*train_data1.shape[0] + ['b']*test_data1.shape[0]
        model1 = svm.SVC(probability= True)
        model1.fit(model_data1, model_label1)

        model_data2 = np.concatenate((train_data2, test_data2), axis = 0)
        model_label2 = ['a']*train_data2.shape[0] + ['b']*test_data2.shape[0]
        model2 = svm.SVC(probability= True)
        model2.fit(model_data2, model_label2)

        p_vals = []
        inf_scores1 = model2.predict_proba(model_data1)
        train_scores1 = inf_scores1[0:train_data1.shape[0], 0]
        for i in range(test_data1.shape[0]):
            cur_inf_score = inf_scores1[train_data1.shape[0]+i, 0]
            cur_p = (np.sum(cur_inf_score >= train_scores1) +1) / (train_data1.shape[0] + 1)
            p_vals.append(float(cur_p))

        inf_scores2 = model1.predict_proba(model_data2)
        train_scores2 = inf_scores2[0:train_data2.shape[0], 0]
        for i in range(test_data2.shape[0]):
            cur_inf_score = inf_scores2[train_data2.shape[0]+i, 0]
            cur_p = (np.sum(cur_inf_score >= train_scores2) +1) / (train_data2.shape[0] + 1)
            p_vals.append(float(cur_p))

        accepted_indices = [i for i,e in enumerate(p_vals) if e >= alpha]
        rejected_indices = [i for i,e in enumerate(p_vals) if e < alpha]

        for i in rejected_indices:
            results[i] = {"Outlier"}
        for i in accepted_indices:
            results[i] = {self.name}
        if self.__should_continue(accepted_indices):
            test_data_next_level = test_data[accepted_indices, :]
            children_results = []
            for c in self.children:
                children_results.append(c.run_inference(test_data_next_level))

            for i in range(len(accepted_indices)):
                org_index = accepted_indices[i]
                all_labels = set()
                children_outliers_count = 0
                for j in range(len(children_results)):
                    child_j_results = children_results[j][i]

                    if child_j_results == {"Outlier"}:
                        children_outliers_count += 1
                    else:
                        all_labels.update(child_j_results)
                if children_outliers_count == len(self.children):
                    results[org_index] = {"Outlier"}
                else:
                    results[org_index] = all_labels

        return results

    # An internal method that decide whether to continue to the next level
    # The criterias are:
    # 1) the current node is not a leaf node
    # 2) the data to be processed is more than a minimal threshold, which is currently set as 30
    def __should_continue(self, data):
        return True if self.is_leaf is not True and len(data) > 30 else False



if __name__ == "__main__":
    # read data
    data = np.loadtxt('../mousebrain/mousebrain_simulate3_pcalarge.txt')
    #data = data[:,1:16]
    #data = data.T
    labels = np.loadtxt('../mousebrain/mousebrain_simlabel3_large.txt',dtype='str')

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

    x_train, x_test, y_train, y_test = train_test_split(x_inlier, y_labels,
                                                        test_size = 0.5, random_state=52)


    x_test_all = np.concatenate((x_test, x_outlier), axis=0)
    y_test_all = np.concatenate((y_test, y_outlier), axis=0)

    tr = BCOPSTree(' (Neuron2, Neuron3,(Neuron1-1,Neuron1-2)Neuron1)Neurons ')
    print(tr)
    tr.attach_data(x_train, y_train)
    results = tr.run_inference(x_test_all)
    print(results)