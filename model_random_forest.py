from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score
from dummy_classifier import dummy_classifier
import numpy as np

# split into train and test, X is features and Y is the label 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# take four parameters, X is the independent variables(features), Y is the dependent varible(label)
def train_random_forest(X_train, X_test, y_train, y_test):
    # train algorithm by using cvSearch's best parameter
    parameters = {
    "n_estimators": np.random.randint(low=5, high=50, size=(5,)),
    "max_depth": np.random.randint(low=5, high=100, size=(5,)),
    "criterion": ["gini", "entropy"]}
    
    regressor = RandomForestClassifier(n_estimators=20, random_state=0)
    
    search = GridSearchCV(regressor, parameters)
    search.fit(X_train,y_train)
    
    print("Best parameters set found on development set:")
    print(search.best_params_)
    print("Best estimator:",search.best_estimator_)
    best_estimator= search.best_estimator_
    y_pred =best_estimator.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    #print("Detailed classification report:")
    #print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test,y_pred)# need to change after this

    macroF1=f1_score(y_test, y_pred, average='macro')
    microF1=f1_score(y_test, y_pred, average='micro')
    weightedF1=f1_score(y_test, y_pred, average='weighted')
    #dummy=dummy_classifier(X_train, y_train)
    return acc, macroF1,microF1,weightedF1
