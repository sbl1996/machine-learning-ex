
from sklearn.model_selection import GridSearchCV
from sklearn import svm

def dataset3_params(X, y):
  values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
  param_grid = [
    {'C': values, 'gamma': values, 'kernel': ['rbf']}
  ]
  clf = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='precision')
  clf.fit(X, y)
  params = clf.best_params_
  return params['C'], params['gamma'], clf.best_estimator_