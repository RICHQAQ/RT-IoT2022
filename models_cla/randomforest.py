import numpy as np
from collections import Counter
from rich.progress import track
from multiprocessing import Pool
from functools import partial
import os

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 分割特征
        self.threshold = threshold  # 分割阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶节点取值


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def entropy(self, y):
        # 优化熵的计算
        hist = np.bincount(y)
        ps = hist / len(y)
        ps = ps[ps > 0]  # 只保留非零概率
        return -(ps * np.log2(ps)).sum()
    
    def information_gain(self, X, y, feature_idx, threshold):
        # 向量化的信息增益计算
        parent_entropy = self.entropy(y)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if not left_mask.any() or not right_mask.any():
            return 0
            
        n = len(y)
        n_l, n_r = left_mask.sum(), right_mask.sum()
        e_l, e_r = self.entropy(y[left_mask]), self.entropy(y[right_mask])
        
        return parent_entropy - (n_l * e_l + n_r * e_r) / n

    def best_split(self, X, y, feature_indices):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feature_indices:
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feat_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)

        # 寻找最佳分割点
        feature_indices = np.arange(n_features)
        feature_idx, threshold = self.best_split(X, y, feature_indices)

        if feature_idx is None:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)

        # 递归构建子树
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        left = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_idx, threshold, left, right)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        return self.predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, random_state=None, n_jobs=-1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state if random_state is not None else np.random.randint(1000)
        self.trees = []

    def _train_tree(self, X, y, seed):
        np.random.seed(seed)
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        tree = DecisionTree(max_depth=self.max_depth)
        tree.fit(X[idxs], y[idxs])
        return tree

    def fit(self, X, y):
        # 保存类别信息
        self.classes_ = np.unique(y)
        
        # 串行训练树
        self.trees = []
        for i in track(range(self.n_trees)):
            seed = self.random_state + i
            tree = self._train_tree(X, y, seed)
            self.trees.append(tree)
            
        return self
        
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([
            Counter(pred).most_common(1)[0][0] 
            for pred in predictions.T
        ])
        
    def score(self, X, y):
        """计算预测准确率"""
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        """获取模型参数"""
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs
        }

    def set_params(self, **parameters):
        """设置模型参数"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self