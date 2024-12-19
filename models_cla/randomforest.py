import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import Parallel, delayed

class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', random_state=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.classes_ = None

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _train_single_tree(self, X, y, seed):
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=seed
        )
        X_sample, y_sample = self._bootstrap_sample(X, y)
        tree.fit(X_sample, y_sample)
        return tree

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        seeds = [self.random_state + i if self.random_state is not None else None for i in range(self.n_estimators)]
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_single_tree)(X, y, seed) for seed in seeds
        )
        return self

    def predict(self, X):
        predictions = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(tree.predict)(X) for tree in self.estimators_
            )
        )
        return np.array([Counter(pred).most_common(1)[0][0] for pred in predictions.T])

    def predict_proba(self, X):
        proba = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(tree.predict_proba)(X) for tree in self.estimators_
            )
        )
        return np.mean(proba, axis=0)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)