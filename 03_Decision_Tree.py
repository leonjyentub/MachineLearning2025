# https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea

import collections
import math
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split


def entropy_func(c, n):
    """
    The math formula
    """
    return -(c*1.0/n)*math.log(c*1.0/n, 2)


def entropy_cal(c1, c2):
    """
    Returns entropy of a group of data
    c1: count of one class
    c2: count of another class
    """
    if c1 == 0 or c2 == 0:  # when there is only one class in the group, entropy is 0
        return 0
    return entropy_func(c1, c1+c2) + entropy_func(c2, c1+c2)

# get the entropy of one big circle showing above


def entropy_of_one_division(division):
    """
    Returns entropy of a divided group of data
    Data may have multiple classes
    """
    s = 0
    n = len(division)
    classes = set(division)
    for c in classes:   # for each class, get entropy
        n_c = sum(division == c)
        e = n_c*1.0/n * entropy_cal(sum(division == c),
                                    sum(division != c))  # weighted avg
        s += e
    return s, n

# The whole entropy of two big circles combined


def get_entropy(y_predict, y_real):
    """
    Returns entropy of a split
    y_predict is the split decision, True/Fasle, and y_true can be multi class
    """
    if len(y_predict) != len(y_real):
        print('They have to be the same length')
        return None
    n = len(y_real)
    s_true, n_true = entropy_of_one_division(
        y_real[y_predict])  # left hand side entropy
    s_false, n_false = entropy_of_one_division(
        y_real[~y_predict])  # right hand side entropy
    # overall entropy, again weighted average
    s = n_true*1.0/n * s_true + n_false*1.0/n * s_false
    return s


class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth

    def fit(self, x, y, par_node={}, depth=0):
        if par_node is None:
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val': y[0], 'count': collections.Counter(y)}
        elif depth >= self.max_depth:
            return None
        else:
            col, cutoff, entropy = self.find_best_split_of_all(x, y)
            # find one split given an information gain
            y_left = y[x[:, col] < cutoff]
            y_right = y[x[:, col] >= cutoff]
            par_node = {'col': features[col],
                        'index_col': col,
                        'cutoff': cutoff,
                        'val': np.round(np.mean(y)),
                        'count': collections.Counter(y)}
            par_node['left'] = self.fit(
                x[x[:, col] < cutoff], y_left, {}, depth+1)
            par_node['right'] = self.fit(
                x[x[:, col] >= cutoff], y_right, {}, depth+1)
            self.depth += 1
            self.trees = par_node
            return par_node

    def find_best_split_of_all(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y)
            if entropy == 0:    # find the first perfect cutoff. Stop Iterating
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy

    def find_best_split(self, col, y):
        min_entropy = 10
        n = len(y)
        for value in sorted(list(set(col))):  # get all unique values
            y_predict = col < value
            my_entropy = get_entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff

    def all_same(self, items):
        return all(x == items[0] for x in items)

    def predict(self, x):
        tree = self.trees
        results = np.array([0]*len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results

    def _get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                if cur_layer['left'] == None:
                    return cur_layer.get('val')
                else:
                    cur_layer = cur_layer['left']
            else:
                if cur_layer['right'] == None:
                    return cur_layer.get('val')
                else:
                    cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')


if __name__ == "__main__":
    iris = datasets.load_iris()
    x = iris.data  # [:, [2, 3]]
    features = iris.feature_names  # [2:4]
    y = iris.target
    # 將數據集分割為訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X_train, y_train)
    pprint(clf.trees)
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(y_test)
    print(
        f"train accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred)*100} %")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['0', '1', '2'])
    disp.plot()
    plt.show()
