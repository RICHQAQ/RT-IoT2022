import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, ClassifierMixin


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_encoded = np.zeros((X.shape[0], len(self.classes_)))
        for i, label in enumerate(self.classes_):
            y_encoded[y == label, i] = 1

        self.estimators_ = []
        y_pred = np.zeros_like(y_encoded)

        for _ in range(self.n_estimators):
            residuals = y_encoded - y_pred
            estimator = DecisionTreeRegressor(
                max_depth=self.max_depth, random_state=self.random_state
            )
            estimator.fit(X, residuals)
            y_pred += self.learning_rate * estimator.predict(X)
            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], len(self.classes_)))
        for estimator in self.estimators_:
            y_pred += self.learning_rate * estimator.predict(X)
        return self.classes_[np.argmax(y_pred, axis=1)]

    def predict_proba(self, X):
        y_pred = np.zeros((X.shape[0], len(self.classes_)))
        for estimator in self.estimators_:
            y_pred += self.learning_rate * estimator.predict(X)
        return self._softmax(y_pred)

    def _softmax(self, X):
        exp_X = np.exp(X)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
