def train_naive_bayes(X_train, X_test, y_train, y_test):

    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    import pandas as pd
    import numpy as np


    gnb=GaussianNB()
    gnb.fit(X_train, y_train)

    # evaluate
    y_pred = gnb.predict(X_test)
    acc = np.sum(y_test == y_pred) / X_test.shape[0]
    print("Test Acc : %.3f" % acc)

    # predict
    y_proba = gnb.predict_proba(X_test[:1])
    print(gnb.predict(X_test[:1]))
    print("The estimated probability:", y_proba)