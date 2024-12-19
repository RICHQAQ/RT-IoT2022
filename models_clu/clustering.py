import numpy as np
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed


class KMeans:
    def __init__(self, n_clusters=3, max_iters=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def _kmeans_plus_plus_init(self, X):
        """使用K-means++初始化聚类中心"""
        n_samples = X.shape[0]

        # 随机选择第一个中心
        centers = [X[np.random.randint(n_samples)]]

        # 选择剩余的中心
        for _ in range(1, self.n_clusters):
            # 计算到最近中心的距离平方
            distances = np.array(
                [min([np.sum((x - c) ** 2) for c in centers]) for x in X]
            )
            # 概率采样
            probs = distances / distances.sum()
            # 选择新中心
            new_center_idx = np.random.choice(n_samples, p=probs)
            centers.append(X[new_center_idx])

        return np.array(centers)

    def _compute_distances_optimized(self, X, centers):
        """优化的距离计算"""
        # 使用矩阵运算加速
        n_samples = X.shape[0]
        n_centers = centers.shape[0]

        # 展开平方项: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        XX = np.sum(X * X, axis=1)[:, np.newaxis]  # shape (n_samples, 1)
        CC = np.sum(centers * centers, axis=1)  # shape (n_centers,)
        XC = np.dot(X, centers.T)  # shape (n_samples, n_centers)

        distances = XX + CC - 2 * XC
        return np.maximum(distances, 0)  # 避免数值误差导致的负值

    def fit_predict(self, X):
        """训练模型并预测"""
        # K-means++初始化
        self.centroids = self._kmeans_plus_plus_init(X)

        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()

            # 优化的距离计算
            distances = self._compute_distances_optimized(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            # 更新中心
            for i in range(self.n_clusters):
                if np.sum(labels == i) > 0:
                    self.centroids[i] = X[labels == i].mean(axis=0)

            # 收敛检查
            if np.allclose(old_centroids, self.centroids, rtol=1e-4):
                break

        return labels


class EM:
    def __init__(
        self,
        n_components=3,
        max_iters=100,
        tol=1e-3,
        n_init=10,
        reg_covar=1e-6,
        random_state=None,
        n_jobs=-1,
    ):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _kmeans_plus_plus_init(self, X):
        """使用k-means++初始化均值"""
        n_samples, n_features = X.shape
        centers = np.empty((self.n_components, n_features))

        # 随机选择第一个中心
        centers[0] = X[np.random.randint(n_samples)]

        # 选择剩余的中心
        for k in range(1, self.n_components):
            # 计算到最近中心的距离平方
            dist = np.min([np.sum((X - c) ** 2, axis=1) for c in centers[:k]], axis=0)
            # 概率采样
            probs = dist / dist.sum()
            centers[k] = X[np.random.choice(n_samples, p=probs)]

        return centers

    def _init_parameters(self, X):
        """初始化模型参数"""
        n_samples, n_features = X.shape

        # 使用k-means++初始化均值
        self.means_ = self._kmeans_plus_plus_init(X)

        # 初始化协方差矩阵
        self.covs_ = np.array(
            [
                np.cov(X.T) + self.reg_covar * np.eye(n_features)
                for _ in range(self.n_components)
            ]
        )

        # 初始化混合权重
        self.weights_ = np.ones(self.n_components) / self.n_components

    def _e_step(self, X):
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            try:
                gaussian = multivariate_normal(
                    mean=self.means_[k],
                    cov=self.covs_[k],
                    allow_singular=True,  # 允许奇异矩阵
                )
                resp[:, k] = self.weights_[k] * gaussian.pdf(X)
            except:
                resp[:, k] = np.finfo(resp.dtype).eps

        # 数值稳定性处理
        resp_sum = resp.sum(axis=1)[:, np.newaxis]
        resp_sum[resp_sum == 0] = 1e-300
        resp /= resp_sum

        return resp

    def _m_step(self, X, resp):
        n_samples, n_features = X.shape

        # 更新权重
        nk = resp.sum(axis=0)
        nk[nk < 1e-6] = 1e-6
        self.weights_ = nk / n_samples

        # 更新均值
        self.means_ = np.dot(resp.T, X) / nk[:, np.newaxis]

        # 更新协方差
        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov = np.dot((resp[:, k : k + 1] * diff).T, diff) / nk[k]

            # 正则化处理
            cov.flat[:: n_features + 1] += self.reg_covar

            # 确保对称性
            cov = (cov + cov.T) / 2
            self.covs_[k] = cov

    def _compute_lower_bound(self, X, resp):
        """改进的下界计算"""
        n_samples = X.shape[0]
        log_prob_norm = 0

        for k in range(self.n_components):
            try:
                # 数值稳定性处理
                weights_k = np.clip(self.weights_[k], 1e-300, None)
                log_weights = np.log(weights_k)
                
                gaussian = multivariate_normal(
                    mean=self.means_[k], 
                    cov=self.covs_[k], 
                    allow_singular=True
                )
                
                # 使用log-sum-exp技巧
                log_probs = log_weights + gaussian.logpdf(X)
                log_probs = np.clip(log_probs, -1e300, None)
                
                # 处理无效值
                with np.errstate(invalid='ignore', divide='ignore'):
                    valid_log_probs = np.where(np.isfinite(log_probs), log_probs, -np.inf)
                    log_prob_norm += np.nansum(valid_log_probs)
                
            except Exception as e:
                print(f"Component {k} failed: {str(e)}")
                continue

        return log_prob_norm

    def _single_fit(self, X):
        """单次EM拟合"""
        self._init_parameters(X)
        lower_bound = -np.inf

        for _ in range(self.max_iters):
            prev_lower_bound = lower_bound

            # E步
            resp = self._e_step(X)

            # M步
            self._m_step(X, resp)

            # 计算下界
            lower_bound = self._compute_lower_bound(X, resp)

            # 收敛检查
            change = lower_bound - prev_lower_bound
            if abs(change) < self.tol:
                break

        return lower_bound, resp

    def fit_predict(self, X):
        """训练模型并预测聚类标签"""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        best_score = -np.inf
        best_resp = None

        for _ in range(self.n_init):
            try:
                self._init_parameters(X)
                lower_bound, resp = self._single_fit(X)

                if lower_bound > best_score:
                    best_score = lower_bound
                    best_resp = resp
            except:
                continue

        if best_resp is None:
            return np.zeros(X.shape[0])

        # 返回最可能的类别
        return np.argmax(best_resp, axis=1)
