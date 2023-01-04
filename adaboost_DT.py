import numpy as np

from sklearn import model_selection, datasets, ensemble, svm, linear_model, tree

X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

# param_grid = {'estimator__criterion': ['gini', 'entropy', 'log_loss'],
#               'estimator__max_depth': [1, 2, 3, 4],
#               'n_estimators': [5, 25, 50, 100, 200, 500],
#               'learning_rate': [0.5, 1, 10, 100]}

# estimator = ensemble.AdaBoostClassifier(estimator=tree.DecisionTreeClassifier(), random_state=0)

# cv_model = model_selection.GridSearchCV(estimator=estimator, param_grid=param_grid, verbose=3, n_jobs=-1)

# cv_model.fit(X_train, y_train)

# print(cv_model.best_params_)

estimator = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)

ab_model = ensemble.AdaBoostClassifier(estimator=estimator, learning_rate=1, n_estimators=50, random_state=0)

ab_model.fit(X_train, y_train)

print(ab_model.score(X_test, y_test))