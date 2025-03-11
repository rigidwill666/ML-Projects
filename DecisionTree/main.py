import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['font.weight'] = 'bold'

def main():
    # 从 CSV 文件中读取数据
    try:
        iris = pd.read_csv('../dataset/Iris.csv')
    except FileNotFoundError:
        print("错误：未找到 'dataset/Iris.csv' 文件，请检查文件路径和文件名!")
        return

    # 提取特征和标签
    X = iris.iloc[:, :-1].values
    y = iris.iloc[:, -1].values

    # 分割数据集为训练集和测试集（70%训练，30%测试）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 初始化决策树分类器
    clf = DecisionTreeClassifier(random_state=42)
    # 训练模型
    clf.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("模型准确率:", accuracy)

    # 输出详细的分类报告
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    feature_names = iris.columns[:-1].tolist()
    class_names = np.unique(y).astype(str).tolist()
    # 调整图形大小和分辨率
    plt.figure(figsize=(4, 4), dpi=200)
    # 自定义字体和颜色
    plot_tree(clf,
              filled=True,
              feature_names=feature_names,
              class_names=class_names,
              fontsize=8,  # 设置字体大小
              node_ids=True,  # 显示节点编号
              impurity=False,  # 不显示基尼指数或熵
              rounded=True,  # 节点边框圆角
              proportion=True,  # 显示样本比例
              precision=2,  # 数值精度
              label='root',  # 只在根节点显示标签
              max_depth=4  # 限制树的深度
              )
    # 添加标题
    plt.title("决策树可视化", fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()
