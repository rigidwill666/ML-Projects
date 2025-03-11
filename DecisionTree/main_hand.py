import numpy as np
import pandas as pd
def gini(y):
    """
    计算数据集 y 的 Gini 不纯度
    """
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)

def best_split(X, y):
    """
    寻找最佳划分特征和阈值，返回 (最佳特征索引, 最佳阈值, 信息增益)
    """
    best_gain = 0
    best_feature, best_threshold = None, None
    parent_impurity = gini(y)
    n_samples, n_features = X.shape

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            # 根据阈值划分数据集
            left_mask = X[:, feature] < threshold
            right_mask = X[:, feature] >= threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_impurity = gini(y[left_mask])
            right_impurity = gini(y[right_mask])
            n_left, n_right = np.sum(left_mask), np.sum(right_mask)

            # 计算信息增益
            gain = parent_impurity - (n_left / n_samples * left_impurity + n_right / n_samples * right_impurity)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

def build_tree(X, y, depth=0, max_depth=3):
    """
    递归构建决策树，达到纯度或最大深度时返回叶节点，
    叶节点中保存的是类别（采用多数表决方式）。
    """
    node = {}

    # 如果所有样本均属于同一类或达到最大深度，则构造叶节点
    if len(np.unique(y)) == 1 or depth == max_depth:
        # 采用多数表决确定叶节点类别
        node['value'] = np.argmax(np.bincount(y))
        return node

    feature, threshold, gain = best_split(X, y)
    # 若无法找到有效的划分则直接构造叶节点
    if gain == 0 or feature is None:
        node['value'] = np.argmax(np.bincount(y))
        return node

    # 划分左右子树
    left_mask = X[:, feature] < threshold
    right_mask = X[:, feature] >= threshold

    node['feature_index'] = feature
    node['threshold'] = threshold
    node['left'] = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth)
    node['right'] = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
    return node

def predict_sample(node, sample):
    """
    根据构建好的树对单个样本进行预测
    """
    if 'value' in node:
        return node['value']
    if sample[node['feature_index']] < node['threshold']:
        return predict_sample(node['left'], sample)
    else:
        return predict_sample(node['right'], sample)

def predict(tree, X):
    """
    对样本集 X 进行预测
    """
    return np.array([predict_sample(tree, sample) for sample in X])

def accuracy(y_true, y_pred):
    """
    计算预测准确率
    """
    return np.sum(y_true == y_pred) / len(y_true)

def print_tree(node, depth=0):
    """
    以文本形式打印决策树结构
    """
    indent = "  " * depth
    if 'value' in node:
        print(indent + "Leaf:", node['value'])
    else:
        print(indent + f"[X{node['feature_index']}] < {node['threshold']:.3f}?")
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)

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
    if isinstance(y[0], str):
        y, _ = pd.factorize(y)

    # 打乱数据集，并划分为训练集（70%）和测试集（30%）
    np.random.seed(42)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    split = int(0.7 * len(y))
    train_indices = indices[:split]
    test_indices = indices[split:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # 构建决策树，设定最大深度为 3
    tree = build_tree(X_train, y_train, max_depth=4)

    # 在测试集上进行预测并计算准确率
    y_pred = predict(tree, X_test)
    acc = accuracy(y_test, y_pred)
    print("准确率:", acc)
    print("测试集标签:", y_test)
    print("预测结果:", y_pred)

    # 打印决策树结构
    print("\n决策树结构:")
    print_tree(tree)

if __name__ == "__main__":
    main()
