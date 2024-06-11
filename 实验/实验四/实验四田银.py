import numpy as np
import pandas as pd
from math import log
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def create_data():
    datasets = [
        ['是', '是', '是'],
        ['是', '是', '是'],
        ['是', '否', '否'],
        ['否', '是', '否'],
        ['否', '否', '否'],
    ]
    labels = ['不浮出水面是否可以生存', '是否有脚蹼', '属于鱼类']
    return datasets, labels


# 计算经验熵
def calc_ent(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(p / data_length) * log(p / data_length, 2) for p in label_count.values()])
    return ent


# 计算条件经验熵
def cond_ent(datasets, axis=0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    cond_ent = sum([(len(p) / data_length) * calc_ent(p) for p in feature_sets.values()])
    return cond_ent


# 计算信息增益
def info_gain(ent, cond_ent):
    return ent - cond_ent


# 计算所有特征的信息增益，并找出决策树根结点
def info_gain_train(datasets, labels):
    count = len(datasets[0]) - 1
    ent = calc_ent(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))
        best_feature.append((c, c_info_gain))
        print('特征({}) 的信息增益为： {:.3f}'.format(labels[c], c_info_gain))
    # 比较大小
    best_ = max(best_feature, key=lambda x: x[-1])
    return best_[0]


# 构建决策树
def build_tree(datasets, labels):
    data = datasets[:]
    class_list = [example[-1] for example in data]
    if class_list.count(class_list[0]) == len(class_list):  # 类别完全相同则停止划分
        return class_list[0]
    if len(data[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类
        return max(class_list, key=class_list.count)

    best_feat = info_gain_train(data, labels)
    best_feat_label = labels[best_feat]
    tree = {best_feat_label: {}}

    del (labels[best_feat])
    feat_values = [example[best_feat] for example in data]
    unique_vals = set(feat_values)

    for value in unique_vals:
        sub_labels = labels[:]  # 子集合需要一个新的标签集合
        sub_data = split_data(data, best_feat, value)
        tree[best_feat_label][value] = build_tree(sub_data, sub_labels)

    return tree


# 按照给定特征划分数据集
def split_data(data, axis, value):
    ret_data = []
    for feat_vec in data:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data.append(reduced_feat_vec)
    return ret_data


# 获取叶节点数目
def get_num_leafs(tree):
    num_leafs = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


# 获取树的层数
def get_tree_depth(tree):
    max_depth = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


# 绘制节点
def plot_node(node_txt, center_pt, parent_pt, node_type):
    arrow_args = dict(arrowstyle="<-")
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


# 标注节点
def plot_mid_text(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


# 绘制树
def plot_tree(tree, parent_pt, node_txt):
    plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    d = {'boxstyle': "sawtooth", 'fc': "0.8"}
    num_leafs = get_num_leafs(tree)
    depth = get_tree_depth(tree)
    first_str = list(tree.keys())[0]
    cntr_pt = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_w, plot_tree.y_off)
    plot_mid_text(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, d)
    second_dict = tree[first_str]
    plot_tree.y_off = plot_tree.y_off - 1.0 / plot_tree.total_d
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.x_off = plot_tree.x_off + 1.0 / plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.x_off, plot_tree.y_off), cntr_pt, d)
            plot_mid_text((plot_tree.x_off, plot_tree.y_off), cntr_pt, str(key))
    plot_tree.y_off = plot_tree.y_off + 1.0 / plot_tree.total_d


# 创建绘制面板
def create_plot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leafs(tree))
    plot_tree.total_d = float(get_tree_depth(tree))
    plot_tree.x_off = -0.5 / plot_tree.total_w
    plot_tree.y_off = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    plt.show()


if __name__ == '__main__':
    data, labels = create_data()
    labels_copy = labels[:]  # 为了不在构建决策树时改变原始标签列表，复制一份
    my_tree = build_tree(data, labels_copy)
    create_plot(my_tree)
