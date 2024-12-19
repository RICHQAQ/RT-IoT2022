import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class SVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel="rbf", gamma="scale", max_iter=1000, tol=1e-3, random_state=None):
        self.C = C  # Regularization parameter
        self.kernel = kernel
        self.gamma = gamma  # RBF kernel parameter
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.alpha = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.b = 0.0

    def _rbf_kernel(self, x1, x2):
        if self.gamma == "scale":
            gamma_value = 1.0 / (x1.shape[1] * x1.var())
        else:
            gamma_value = self.gamma
        dist = (
            np.sum(x1**2, axis=1).reshape(-1, 1)
            + np.sum(x2**2, axis=1)
            - 2 * np.dot(x1, x2.T)
        )
        return np.exp(-gamma_value * dist)

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        y = np.where(y == self.classes_[0], -1, 1)

        K = self._rbf_kernel(X, X)
        self.alpha = np.zeros(n_samples)
        b = 0.0

        for iteration in range(self.max_iter):
            alpha_prev = self.alpha.copy()
            for i in range(n_samples):
                error_i = (
                    np.sum(self.alpha * y * K[i]) + b - y[i]
                )
                for j in range(i + 1, n_samples):
                    error_j = (
                        np.sum(self.alpha * y * K[j]) + b - y[j]
                    )
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    L = max(0, alpha_j_old - alpha_i_old if y[i] != y[j] else alpha_i_old + alpha_j_old - self.C)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old if y[i] != y[j] else alpha_i_old + alpha_j_old)

                    self.alpha[j] -= y[j] * (error_i - error_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

            b_candidates = (
                y - np.sum(self.alpha * y * K, axis=1)
            )
            b = np.mean(b_candidates)

            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break

        sv = self.alpha > 1e-5
        self.support_vectors_ = X[sv]
        self.support_vector_labels_ = y[sv]
        self.alpha = self.alpha[sv]

        self.b = np.mean(
            self.support_vector_labels_
            - np.sum(
                self.alpha
                * self.support_vector_labels_
                * self._rbf_kernel(self.support_vectors_, self.support_vectors_),
                axis=0,
            )
        )

        return self

    def predict(self, X):
        kernel = self._rbf_kernel(X, self.support_vectors_)
        decision = (
            np.sum(self.alpha * self.support_vector_labels_ * kernel, axis=1) + self.b
        )
        return np.where(decision < 0, self.classes_[0], self.classes_[1])

    def predict_proba(self, X):
        kernel = self._rbf_kernel(X, self.support_vectors_)
        decision = (
            np.sum(self.alpha * self.support_vector_labels_ * kernel, axis=1) + self.b
        )
        proba = 1 / (1 + np.exp(-decision))
        return np.vstack([1 - proba, proba]).T
