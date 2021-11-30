from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np
from dummy_classifier import dummy_classifier
from sklearn.metrics import f1_score
def train_linear_regression(X_train, X_test, y_train, y_test):
    #x should be the 2 dimensional array, y should be column or 2d array
    # this is just sample code, because we don't have y right now
    lg_model = linear_model.LinearRegression()
    #let the linear regression fit into x and y
    lg_model.fit(X_train,y_train)
    y_pred = lg_model.predict(X_test)
    
    
    #round the labels which in real numbers
    y_pred_round = np.round(y_pred)
    #accuracy method 1: really high accuracy 80-90
    #accuracyRounded = lg_model.score(X_test, y_pred_round)
    #accuracy method 2: really low accuracy 9 -13
    acc = accuracy_score(y_test, y_pred_round)
    
    macroF1=f1_score(y_test, y_pred_round, average='macro')
    microF1=f1_score(y_test, y_pred_round, average='micro')
    weightedF1=f1_score(y_test, y_pred_round, average='weighted')
    #dummy=dummy_classifier(X_train, y_train)
    return acc, macroF1,microF1,weightedF1
   



