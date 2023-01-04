import numpy as np

from sklearn import model_selection, datasets, ensemble, svm, linear_model

X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

param_grid = {"estimator__degree" : [2, 3, 4, 5, 6, 7],
              "estimator__kernel" :   ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
              "n_estimators": [50, 100, 250, 500, 1000, 2000]}

estimator = ensemble.AdaBoostClassifier(estimator=svm.NuSVC(), algorithm='SAMME', random_state=0)

cv_model = model_selection.GridSearchCV(estimator=estimator, param_grid=param_grid, verbose=3)

cv_model.fit(X_train, y_train)

# param_grid = {'penalty': ['l1', 'l2', 'elasticnet', None]}
# cv_model = model_selection.GridSearchCV(estimator=linear_model.LogisticRegression(), param_grid=param_grid, verbose=3)

print(cv_model.best_params_)

# ab_model = ensemble.AdaBoostClassifier(estimator=svm.NuSVC(), algorithm='SAMME', random_state=0)

# ab_model.fit(X_train, y_train)

# print(ab_model.score(X_test, y_test))
