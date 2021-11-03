def linear_regression(param_x,param_y):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from feature_lb.py import df
    #x should be the 2 dimensional array, y should be column or 2d array
    # this is just sample code, because we don't have y right now
    x = np.array(param_x).reshape((-1, 1))
    y = np.array(param_y)
    model = LinearRegression
    #let the linear regression fit into x and y
    model.fit(x,y)
    
    #get results
    #get coefficient of determination, R^2
    lin_r_square = model.score(x, y)
    lin_intercept = model.intercept_
    lin_coefficient = model.coef_
    print('coefficient of determination:', lin_r_square)
    print('intercept:', lin_intercept)
    print('slope:', lin_coefficient)

    #predict response
    #Once there is a satisfactory model, 
    #use it for predictions with either existing or new data.
    y_pred = model.predict(x)
    #or using this:
    #y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)
    print('predicted response:', y_pred, sep='\n')
    
    #apply model to new data
    x_new = np.arange(10).reshape((-1, 2))
    y_new = model.predict(x_new)
    print(x_new)
    print(y_new)