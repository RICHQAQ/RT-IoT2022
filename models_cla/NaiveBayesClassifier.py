import math
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.class_probs = None
        self.class_stats = None

    def gaussian_pdf(self, x, mean, std_dev):
        """计算高斯分布的概率密度函数"""
        if std_dev == 0:
            return 1.0 if x == mean else 0.0
        exponent = math.exp(-0.5 * ((x - mean) ** 2 / (std_dev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * std_dev)) * exponent

    def calculate_class_probabilities(self, y_train):
        """计算每个类的先验概率"""
        class_probs = {}
        total_samples = len(y_train)
        for label in y_train:
            if label not in class_probs:
                class_probs[label] = 0
            class_probs[label] += 1
        for label in class_probs:
            class_probs[label] /= total_samples
        return class_probs

    def calculate_class_statistics(self, X_train, y_train):
        """计算每个类的特征统计量（均值和标准差）"""
        class_stats = {}
        unique_classes = set(y_train)
        for label in unique_classes:
            class_samples = [X_train[i] for i in range(len(y_train)) if y_train[i] == label]
            class_stats[label] = []
            for feature_index in range(len(X_train[0])):
                feature_values = [sample[feature_index] for sample in class_samples]
                mean = sum(feature_values) / len(feature_values)
                std_dev = math.sqrt(sum([(x - mean) ** 2 for x in feature_values]) / len(feature_values))
                class_stats[label].append((mean, std_dev))
        return class_stats

    def fit(self, X, y):
        """训练模型"""
        X = np.array(X)
        y = np.array(y)
        self.class_probs = self.calculate_class_probabilities(y)
        self.class_stats = self.calculate_class_statistics(X, y)
        return self

    def predict(self, X):
        """预测新样本的类别"""
        X = np.array(X)
        predictions = []
        for sample in X:
            class_probabilities = {}
            for label in self.class_probs:
                likelihood = 1
                for feature_index in range(len(sample)):
                    mean, std_dev = self.class_stats[label][feature_index]
                    likelihood *= self.gaussian_pdf(sample[feature_index], mean, std_dev)
                posterior = likelihood * self.class_probs[label]
                class_probabilities[label] = posterior
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(predicted_class)
        return predictions

    def score(self, X, y):
        """符合 sklearn 接口的 score 方法"""
        return np.mean(self.predict(X) == y)
