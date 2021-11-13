import pandas as pd
from feature_lb import feature_lb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

# for standardize data
from sklearn.preprocessing import StandardScaler
# import PCA
from sklearn.decomposition import PCA
from linear_regression import train_linear_regression
from decision_tree import train_decision_tree
from naive_bayes import train_naive_bayes
from random_forest import train_random_forest
# call this function to normalize the data (scalling) and use PCA to reduce dimension(features) to 300 
# take X as features and Y as label (unnormailzied)

def PCA_reduce_dimension(X, Y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=300)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents)
    finalDf = pd.concat([principalDf, Y], axis = 1)
    #print(finalDf)
    return finalDf

def get_data():
    df = feature_lb()
    X = df.loc[:, df.columns != 'New Label']
    y = df['New Label']
    final_df = PCA_reduce_dimension(X,y)
    #print(df)
    return final_df

def fiveCrossValidation(model_name):
    k = 5 # 5 fold
    kf = KFold(n_splits=k, random_state=None)
    acc_total = []
    mse_total = []
    mae_total = []
    rmse_total = []
    r2Score_total = []
    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
        
        if model_name == 'linear_regression':
            acc,mse,mae,rmse = train_linear_regression(X_train, X_test, y_train, y_test)
        elif model_name == 'decision_tree':
            acc,mse,mae,rmse = train_decision_tree(X_train, X_test, y_train, y_test)
        elif model_name == 'naive_bayes':
            acc,mse,mae,rmse = train_naive_bayes(X_train, X_test, y_train, y_test)
        elif model_name == 'random_forest':
            acc,mse,mae,rmse,r2Score = train_random_forest(X_train, X_test, y_train, y_test)
            
        #acc = train_linear_regression(X_train, X_test, y_train, y_test)
        acc_total.append(acc)
        mse_total.append(mse)
        mae_total.append(mae)
        rmse_total.append(rmse)
        r2Score_total.append(r2Score)
        
    avg_acc = sum(acc_total)/k
    avg_mse = sum(mse_total)/k
    avg_mae = sum(mae_total)/k
    avg_rmse = sum(rmse_total)/k
    avg_r2Score = sum(r2Score_total)/k 
    #print('accuracy of each fold - {}'.format(acc_total))
    print("This model is :", model_name)
    print('Avg accuracy : {}'.format(avg_acc))
    print('Avg mse : {}'.format(avg_mse))
    print('Avg mae : {}'.format(avg_mae))
    print('Avg rmse : {}'.format(avg_rmse))
    print('Avg r2 score : {}'.format(avg_r2Score))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get the features
    #df = get_data()
    # 1- Split data to train and test
    df = get_data()
    #X is whole dataframe except label column 
    X = df.loc[:, df.columns != 'New Label']
    #y is label column
    y = df['New Label']
    
    ####train model by using 5 fold cross validation####
    # 2- Cross-fold validation on the train data   
    model_name = "random_forest"
    fiveCrossValidation(model_name)    
    
    
    
    
   

