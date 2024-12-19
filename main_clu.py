import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rich.progress import track
from models_clu import EM
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像时 负号'-' 显示为□和报错的问题。
os.makedirs("output", exist_ok=True)

def evaluate_clustering(name, model, X, n_clusters=None):
    os.makedirs(f"output/{name}", exist_ok=True)
    print(f"\n评估 {name} 聚类效果...")
    # 转换为numpy数组
    X_array = X.to_numpy(np.float64) if isinstance(X, pd.DataFrame) else X
    # 训练模型
    labels = model.fit_predict(X_array)
    
    # 计算评估指标
    if len(np.unique(labels[labels != -1])) > 1:  # 确保有效聚类数 > 1
        sil_score = silhouette_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
    else:
        sil_score = ch_score = db_score = 0
        
    # 可视化聚类结果
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_array)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f"{name} 聚类结果 (PCA降维)")
    plt.xlabel("第一主成分")
    plt.ylabel("第二主成分")
    plt.savefig(f"output/{name}/clustering_result.png")
    plt.close()

    return {
        'name': name,
        'silhouette': sil_score,
        'calinski_harabasz': ch_score,
        'davies_bouldin': db_score,
        'n_clusters': len(np.unique(labels[labels != -1]))
    }

def main():
    # 加载数据
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    data = pd.concat([train_data, test_data])
    
    # 数据预处理 - 分层采样
    target_size = 5000
    grouped = data.groupby('Attack_type')
    sample_sizes = (grouped.size() / len(data) * target_size).astype(int)
    
    # 确保每类至少有一个样本
    sample_sizes = sample_sizes.clip(lower=1)
    
    # 按比例采样
    sampled_data = pd.concat([
        group.sample(n=sample_sizes[name], random_state=42) 
        for name, group in grouped
    ])
    
    n_clusters = len(sampled_data['Attack_type'].unique())
    print(f"聚类个数: {n_clusters}")
    print(f"采样后数据量: {len(sampled_data)}")
    
    X = sampled_data.drop("Attack_type", axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 初始化聚类模型
    models = {
        # 'KMeans': KMeans(n_clusters=n_clusters, random_state=42),
        # 'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        # 'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
        'GaussianMixture': GaussianMixture(n_components=n_clusters, random_state=42),
        'EM': EM(n_components=n_clusters, random_state=42)
    }

    # 评估模型
    results = []
    for name, model in track(models.items(), description="评估聚类模型"):
        result = evaluate_clustering(name, model, X_scaled)
        results.append(result)

    # 生成评估报告
    with open("output/clustering_report.txt", "w", encoding="utf-8") as f:
        f.write("聚类模型评估报告\n")
        f.write("=" * 50 + "\n\n")
        for result in results:
            f.write(f"模型: {result['name']}\n")
            f.write(f"聚类数量: {result['n_clusters']}\n")
            f.write(f"轮廓系数: {result['silhouette']:.4f}\n")
            f.write(f"Calinski-Harabasz指数: {result['calinski_harabasz']:.4f}\n")
            f.write(f"Davies-Bouldin指数: {result['davies_bouldin']:.4f}\n")
            f.write("-" * 50 + "\n")

    # 绘制评估指标对比图
    """
        # 聚类评估指标说明

        ## 轮廓系数 (Silhouette Score)
        - 衡量样本在自己所在簇的紧密程度与其他簇的分离程度
        - 取值范围: [-1, 1]
        - 分数越接近1表示聚类效果越好
        - 分数越接近-1表示可能被分配到错误的簇
        - 0附近表示簇之间有重叠

        ## Calinski-Harabasz指数
        - 又称为方差比准则(VRC)
        - 计算簇间离散度与簇内离散度的比值
        - 取值范围: [0, +∞)
        - 分数越高表示聚类越密集且簇间分离度越好
        - 适用于评估凸形簇

        ## Davies-Bouldin指数
        - 评估簇内样本的平均相似度与簇间样本的相似度比值
        - 取值范围: [0, +∞)
        - 分数越小表示聚类效果越好
        - 对噪声较为敏感
    """
    metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar([r['name'] for r in results], [r[metric] for r in results])
        plt.title(f"{metric}指标对比")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"output/{metric}_comparison.png")
        plt.close()

    print("\n聚类评估完成!")
    print("详细报告已保存至 output/clustering_report.txt")

if __name__ == "__main__":
    main()