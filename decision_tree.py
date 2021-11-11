import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree

def train_decision_tree(X_train, X_test, y_train, y_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtc.score(X_test, y_test)

    y_pred = dtc.predict(X_test)

    print ("Accuracy is :", accuracy_score(y_test,y_pred))

    """results = confusion_matrix(y_test, y_pred) 
    print ('Confusion Matrix :')
    print(results)"""