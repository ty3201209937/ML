import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, value=None, result=None):
        self.feature = feature  # 划分特征
        self.value = value      # 划分特征取值
        self.result = result    # 结果，如果是叶子节点则为类别，否则为None
        self.children = {}      # 子节点字典，key为特征取值，value为子节点

def entropy(data):
    """
    计算数据集的信息熵
    """
    label_counts = Counter(data[:, -1])
    entropy = 0
    for label in label_counts:
        prob = label_counts[label] / len(data)
        entropy -= prob * np.log2(prob)
    return entropy

def split_data(data, feature_index):
    """
    根据特征划分数据集
    """
    feature_values = set(data[:, feature_index])
    split_data_dict = {}
    for value in feature_values:
        split_data_dict[value] = data[data[:, feature_index] == value]
    return split_data_dict

def information_gain(data, feature_index):
    """
    计算信息增益
    """
    base_entropy = entropy(data)
    split_data_dict = split_data(data, feature_index)
    new_entropy = sum((len(split_data_dict[value]) / len(data)) * entropy(split_data_dict[value]) for value in split_data_dict)
    return base_entropy - new_entropy


def select_best_feature(data):
    """
    选择最佳划分特征
    """
    num_features = data.shape[1] - 1
    best_feature_index = -1
    best_info_gain = 0
    for i in range(num_features):
        info_gain = information_gain(data, i)
        print(f"Feature {i}: Information Gain = {info_gain}")
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature_index = i
    print(f"Best Feature Index: {best_feature_index}")

    # 如果最佳特征的信息增益为0，则返回-1
    if best_info_gain == 0:
        return -1

    return best_feature_index


def create_decision_tree(data, labels):
    """
    构建决策树
    """
    # 如果所有实例属于同一类别，则返回该类别
    if len(set(data[:, -1])) == 1:
        return Node(result=data[0, -1])

    # 如果没有更多特征可用，则返回实例数最多的类别
    if data.shape[1] == 1:
        most_common_label = Counter(data[:, -1]).most_common(1)[0][0]
        return Node(result=most_common_label)

    best_feature_index = select_best_feature(data)
    best_feature_label = labels[best_feature_index]

    # 根据最佳特征划分数据集
    split_data_dict = split_data(data, best_feature_index)

    # 删除已选择的特征
    sub_labels = labels[:best_feature_index] + labels[best_feature_index+1:]

    # 递归构建子树
    node = Node(feature=best_feature_label)
    for value, sub_data in split_data_dict.items():
        node.children[value] = create_decision_tree(sub_data, sub_labels)
    return node

def predict(node, sample):
    """
    使用决策树进行分类预测
    """
    if node.result is not None:
        return node.result
    feature_value = sample[node.feature]
    if feature_value not in node.children:
        # 如果测试数据中的特征值在训练集中未出现，则返回None
        return None
    return predict(node.children[feature_value], sample)

# 测试数据
data = np.array([
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
    ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
    ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
    ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
    ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
    ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
    ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
    ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
    ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
    ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
    ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
    ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']
])

labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']

# 构建决策树
decision_tree = create_decision_tree(data, labels)

# 测试样例
test_sample = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑']
predicted_label = predict(decision_tree, test_sample)
print("预测结果：", predicted_label)
