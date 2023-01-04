import numpy as np

from sklearn import model_selection, datasets, ensemble, svm, linear_model, neighbors

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

ab_model = ensemble.AdaBoostRegressor(estimator=svm.SVR(), n_estimators=100, random_state=0)

ab_model.fit(X_train, y_train)

print(ab_model.score(X_test, y_test))

# param_grid = {"estimator__degree" : [2, 3, 4, 5, 6, 7],
#               "estimator__kernel" :   ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
#               "n_estimators": [50, 100, 250, 500, 1000, 2000]}

        
