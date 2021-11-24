import numpy as np
from sklearn.dummy import DummyClassifier

def dummy_classifier(X_train, y_train):
    dummy_clf = DummyClassifier(strategy="stratified", random_state=42)
    dummy_clf.fit(X_train, y_train)
    dummy_clf.predict(X_train)
    print("The scores of dummy classifier: ")
    print(dummy_clf.score(X_train, y_train))