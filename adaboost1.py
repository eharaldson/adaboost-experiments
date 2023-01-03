import numpy as np

from sklearn import model_selection, datasets, ensemble, svm

X, y = datasets.load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

ab_model = ensemble.AdaBoostClassifier(estimator=svm.NuSVC(), algorithm='SAMME', n_estimators=20, random_state=0)

ab_model.fit(X_train, y_train)

print(ab_model.score(X_test, y_test))
