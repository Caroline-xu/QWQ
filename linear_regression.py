from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from matplotlib import pyplot as plt
import numpy as np
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
    accuracyRounded = lg_model.score(X_test, y_pred_round)
    #accuracy method 2: really low accuracy 9 -13
    accuracyScore = accuracy_score(y_test, y_pred_round)
    #print("accuracy score of Linear regression:", accuracyScore)
    #get MSE mean squared error
    mse = mean_squared_error(y_test,y_pred_round)
    #print("mean squared error of Linear regression:", mse)
    #get MAE
    mae = mean_absolute_error(y_test, y_pred_round)
    #print("mean absolute error of Linear regression:", mae)
    #get RMSE
    rmse = np.sqrt(mae)
    #print("root mean squared error of Linear regression:", rmse) 
    
    return accuracyScore,mse,mae,rmse
''' 
    plt.scatter(X_train_reshape, y_pred_round,color = "black")
    plt.plot(X_train_reshape, y_pred_round, color="blue", linewidth=3) 
    plt.title("linear regression's graph")
    plt.xlabel("True values")
    plt.ylabel("Predictions")
    plt.xticks(())
    plt.yticks(())
    plt.show()'''
    
    
    
    
