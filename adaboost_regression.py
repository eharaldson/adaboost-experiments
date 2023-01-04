import numpy as np

from sklearn import model_selection, datasets, ensemble, svm, linear_model, neighbors

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

param_grid = {"estimator__degree" : [2, 3, 4, 5, 6, 7],
              "estimator__kernel" :   ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
              "estimator__nu": [0.3, 0.4, 0.5, 0.6, 0.7],
              "n_estimators": [50, 100, 250, 500, 1000, 2000]}

cv_model = model_selection.GridSearchCV(estimator=ensemble.AdaBoostRegressor(estimator=svm.NuSVR(), random_state=0), n_jobs=-1, verbose=3, param_grid=param_grid)
cv_model.fit(X_train, y_train)