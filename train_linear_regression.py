from sklearn import linear_model
from matplotlib import pyplot as plt

def train_linear_regression(X_train, X_test, y_train, y_test):
    #x should be the 2 dimensional array, y should be column or 2d array
    # this is just sample code, because we don't have y right now
    lg_model = linear_model.LinearRegression()
    #let the linear regression fit into x and y
    lg_model.fit(X_train,y_train)
    predictions = lg_model.predict(X_test)
    
    #plot the line / model
    plt.scatter(y_test, predictions)
    plt.xlabel("True values")
    plt.ylabel("Predictions")
    accuracy = lg_model.score(X_test, y_test)
    print("Accuracy of Linear regression:", accuracy)
    return accuracy
