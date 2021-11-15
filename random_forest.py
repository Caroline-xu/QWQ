from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
# split into train and test, X is features and Y is the label 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# take four parameters, X is the independent variables(features), Y is the dependent varible(label)
def train_random_forest(X_train, X_test, y_train, y_test):
    # train algorithm
    regressor = RandomForestClassifier(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    # evaluating the accuracy 
    # print(confusion_matrix(y_test,y_pred))
    # print(classification_report(y_test,y_pred))
    
    acc = accuracy_score(y_test,y_pred)# need to change after this
    acc = regressor.score(X_test, y_pred)
    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mae)
    from sklearn.metrics import r2_score
    r2Score=r2_score(y_test, y_pred)
    return acc, mse, mae, rmse,r2Score
