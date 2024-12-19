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
    def __init__(self, n_components=3, max_iters=100, tol=1e-3, random_state=None):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
    def _init_params(self, X):
        n_samples, n_features = X.shape
        
        # 使用K-means++初始化均值
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
        labels = kmeans.fit_predict(X)
        
        # 初始化参数
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = np.array([X[labels == k].mean(axis=0) for k in range(self.n_components)])
        self.covs_ = np.array([np.cov(X[labels == k].T) if np.sum(labels == k) > 1 
                              else np.eye(n_features) for k in range(self.n_components)])
                              
    def _e_step(self, X):
        """计算责任度"""
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # 计算多元高斯分布
            try:
                rv = multivariate_normal(self.means_[k], self.covs_[k], allow_singular=True)
                resp[:, k] = self.weights_[k] * rv.pdf(X)
            except:
                resp[:, k] = np.finfo(resp.dtype).eps
                
        # 归一化
        resp_sum = resp.sum(axis=1)[:, np.newaxis]
        resp_sum[resp_sum == 0] = 1e-300
        resp /= resp_sum
        return resp
        
    def _m_step(self, X, resp):
        """更新参数"""
        n_samples = X.shape[0]
        
        # 计算有效样本权重
        nk = resp.sum(axis=0)  # shape: (n_components,)
        mask = nk > 1e-10  # 避免除零
        
        # 更新权重
        self.weights_ = np.where(mask, nk / n_samples, 1e-10)
        
        # 更新均值
        self.means_ = np.zeros_like(self.means_)
        for k in range(self.n_components):
            if mask[k]:
                self.means_[k] = np.dot(resp[:, k], X) / nk[k]
            else:
                self.means_[k] = X.mean(axis=0)  # 退化为整体均值
        
        # 更新协方差
        for k in range(self.n_components):
            if mask[k]:
                diff = X - self.means_[k]
                self.covs_[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
                # 添加正则化项保证正定性
                self.covs_[k] += 1e-6 * np.eye(X.shape[1])
            else:
                self.covs_[k] = np.eye(X.shape[1])  # 退化为单位矩阵
            
    def fit_predict(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # 初始化参数
        self._init_params(X)
        prev_ll = -np.inf
        
        for _ in range(self.max_iters):
            # E步
            resp = self._e_step(X)
            
            # M步
            self._m_step(X, resp)
            
            # 计算对数似然
            ll = self._compute_log_likelihood(X)
            
            # 收敛检查
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
            
        return np.argmax(resp, axis=1)
        
    def _compute_log_likelihood(self, X):
        """计算对数似然"""
        n_samples = X.shape[0]
        ll = 0
        
        for k in range(self.n_components):
            try:
                rv = multivariate_normal(self.means_[k], self.covs_[k], allow_singular=True)
                ll += np.sum(np.log(self.weights_[k] * rv.pdf(X) + 1e-300))
            except:
                continue
                
        return ll / n_samples
