import sklearn.tree
import sklearn.preprocessing
import numpy as np

def generate_tree(X, Y):
    tree = sklearn.tree.DecisionTreeClassifier(max_depth=10)
    X = np.array(X, dtype=object)
    Y = np.array(Y)
    tree.fit(X,Y)
    return tree

# sklearn.tree.DecisionTreeClassifier
# def generate_sets(headlines, known_words):
