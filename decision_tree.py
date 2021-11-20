import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
def train_decision_tree(X_train, X_test, y_train, y_test):
    
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtc.score(X_test, y_test)

    y_pred = dtc.predict(X_test)

    #print ("Accuracy is :", accuracy_score(y_test,y_pred))
    acc = accuracy_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mae)
    return acc, mse, mae, rmse
     
    """results = confusion_matrix(y_test, y_pred) 
    print ('Confusion Matrix :')
    print(results)"""
