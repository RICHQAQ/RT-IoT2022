import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from scipy import linalg

class EM:
    def __init__(self, n_components=3, max_iters=100, tol=1e-3, 
                 reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _check_parameters(self, X):
        X = check_array(X)
        if self.n_components < 1:
            raise ValueError("Invalid number of components")
        return X

    def _init_params(self, X):
        n_samples, n_features = X.shape

        # 使用KMeans初始化聚类中心
        kmeans = KMeans(n_clusters=self.n_components, 
                       random_state=self.random_state)
        labels = kmeans.fit_predict(X)

        # 初始化参数
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = np.array([X[labels == k].mean(axis=0) 
                              for k in range(self.n_components)])

        self.covariances_ = []
        for k in range(self.n_components):
            if np.sum(labels == k) > 1:
                cov = np.cov(X[labels == k].T)
            else:
                cov = np.eye(n_features)
            cov.flat[::n_features + 1] += self.reg_covar
            self.covariances_.append(cov)
        self.covariances_ = np.array(self.covariances_)

        self.converged_ = False
        self.n_iter_ = 0

    def _m_step(self, X, resp):
        """M步:更新模型参数"""
        n_samples, n_features = X.shape

        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps

        self.weights_ = nk / n_samples
        self.means_ = np.dot(resp.T, X) / nk[:, np.newaxis]

        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
            self.covariances_[k].flat[::n_features + 1] += self.reg_covar

    def _e_step(self, X):
        n_samples, n_features = X.shape
        resp = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            try:
                precision = linalg.inv(self.covariances_[k])
                diff = X - self.means_[k]
                log_det = np.log(linalg.det(self.covariances_[k]) + self.reg_covar)

                log_prob = -0.5 * (
                    np.sum(np.dot(diff, precision) * diff, axis=1) +
                    n_features * np.log(2 * np.pi) + log_det
                )
                resp[:, k] = self.weights_[k] * np.exp(log_prob)
            except linalg.LinAlgError:
                resp[:, k] = 0

        resp_sum = resp.sum(axis=1)[:, np.newaxis]
        resp_sum[resp_sum == 0] = 1e-300
        resp /= resp_sum

        return resp

    def _compute_log_likelihood(self, X):
        """计算对数似然"""
        n_samples, n_features = X.shape
        log_likelihood = 0

        for k in range(self.n_components):
            try:
                precision = linalg.inv(self.covariances_[k])
                diff = X - self.means_[k]
                log_det = np.log(linalg.det(self.covariances_[k]) + self.reg_covar)

                log_prob = -0.5 * (
                    np.sum(np.dot(diff, precision) * diff, axis=1) +
                    n_features * np.log(2 * np.pi) + log_det
                )
                log_likelihood += self.weights_[k] * np.exp(log_prob)
            except linalg.LinAlgError:
                continue

        log_likelihood = np.log(log_likelihood + 1e-300).mean()
        return log_likelihood

    def fit(self, X):
        X = self._check_parameters(X)
        self._init_params(X)

        prev_ll = -np.inf
        for n_iter in range(self.max_iters):
            self.n_iter_ = n_iter + 1

            resp = self._e_step(X)
            self._m_step(X, resp)

            ll = self._compute_log_likelihood(X)
            if abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                break

            prev_ll = ll

        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = check_array(X)
        return self._e_step(X)

    def fit_predict(self, X):
        """
        执行拟合并返回预测标签

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            训练数据

        Returns
        -------
        labels : array-like of shape (n_samples,)
            每个样本的聚类标签
        """
        self.fit(X)
        return self.predict(X)
