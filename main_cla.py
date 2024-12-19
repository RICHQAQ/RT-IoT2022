# 构建分类模型
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from models_cla.GradientBoosting import GradientBoostingClassifier
from models_cla import RandomForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from concurrent.futures import ProcessPoolExecutor, as_completed

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像时 负号'-' 显示为□和报错的问题。
os.makedirs("output", exist_ok=True)


def train_and_evaluate(name, model, X_train_scaled, y_train, X_val_scaled, y_test):
    os.makedirs(f"output/{name}", exist_ok=True)
    print(f"\n训练 {name} 并绘制学习曲线...")

    # # 绘制学习曲线并获取训练集和验证集的得分
    # train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
    #     model,
    #     X_train_scaled,
    #     y_train,
    #     cv=5,
    #     n_jobs=2,
    #     train_sizes=np.linspace(0.2, 1.0, 5),
    #     return_times=True,
    # )
    # train_scores_mean = np.mean(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)

    # plt.figure()
    # plt.plot(train_sizes, train_scores_mean, "o-", label="训练得分")
    # plt.plot(train_sizes, test_scores_mean, "o-", label="验证得分")
    # plt.title(f"{name} 学习曲线")
    # plt.xlabel("训练样本数")
    # plt.ylabel("得分")
    # plt.legend(loc="best")
    # plt.savefig(f"output/{name}/learning_curve.png")
    # plt.close()

    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_test, val_pred)
    print(f"{name} 验证集准确率: {accuracy:.4f}")
    print(
        f"{name} 分类报告:\n", classification_report(y_test, val_pred, zero_division=0)
    )
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, val_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} 混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.savefig(f"output/{name}/confusion_matrix.png")
    plt.close()

    return name, accuracy


def main():
    # 加载数据
    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    # 准备特征和标签
    X_train = train_data.drop("Attack_type", axis=1)
    y_train = train_data["Attack_type"]
    X_test = test_data.drop("Attack_type", axis=1)
    y_test = test_data["Attack_type"]

    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_test)

    # 初始化模型
    non_ai_models = {
        # 'RandomForest': RandomForestClassifier(n_estimators=100),
        # "skRandomForest": RandomForestClassifier(n_estimators=10, random_state=42),
        "RandomForest": RandomForest(n_trees=5, random_state=42),
        # "GradientBoosting": GradientBoostingClassifier(
        #     n_estimators=10, random_state=42
        # ),
        # 'SVC': SVC(kernel='linear', random_state=42),
    }

    ai_models = {}

    # 训练和评估模型
    results = {}

    # 训练非AI模型
    for name, model in non_ai_models.items():
        name, accuracy = train_and_evaluate(name, model, X_train_scaled, y_train, X_val_scaled, y_test)
        results[name] = accuracy

    # 绘制模型性能对比图
    plt.figure(figsize=(10, 6))
    plt.plot(results.keys(), results.values())
    plt.title("各模型性能对比")
    plt.xlabel("模型")
    plt.ylabel("准确率")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/model_comparison.png")
    plt.close()

    # 生成总结报告
    best_model = max(results.items(), key=lambda x: x[1])
    with open("output/summary_report.txt", "w", encoding="utf-8") as f:
        f.write("模型训练评估总结报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"评估模型数量: {len(results)}\n")
        f.write(f"最佳模型: {best_model[0]}\n")
        f.write(f"最佳准确率: {best_model[1]:.4f}\n\n")
        f.write("所有模型性能:\n")
        for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{model}: {acc:.4f}\n")

    print("\n训练评估完成!")
    print(f"最佳模型: {best_model[0]}，准确率: {best_model[1]:.4f}")
    print("详细报告已保存至 output/summary_report.txt")


if __name__ == "__main__":
    main()
