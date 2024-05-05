import numpy as np
import pandas as pd
from math import log
def create_data():
    datasets = [['青年', '否',  '一般', '否'],
               ['青年', '否',  '好', '否'],
               ['青年', '是',  '好', '是'],
               ['青年', '否',  '一般', '否'],
               ['中年', '否',  '一般', '否'],
               ['中年', '否', '好', '否'],
               ['老年', '是', '好', '是'],
               ['老年', '是', '非常好', '是'],
               ['老年', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'信贷情况', u'类别']
    return datasets, labels

datasets, labels = create_data()
train_data = pd.DataFrame(datasets, columns=labels)
print(train_data)

#计算经验熵
def calc_ent(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(p / data_length) * log(p / data_length, 2)
                for p in label_count.values()])
    return ent

#计算条件经验熵
def cond_ent(datasets, axis=0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    cond_ent = sum([(len(p) / data_length) * calc_ent(p)
                    for p in feature_sets.values()])
    return cond_ent

#计算信息增益
def info_gain(ent, cond_ent):
    return ent - cond_ent

#计算所有特征的信息增益，并找出决策树根结点
def info_gain_train(datasets):
    count = len(datasets[0]) - 1
    ent = calc_ent(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))
        print(' ',cond_ent(datasets, axis=c))
        best_feature.append((c, c_info_gain))
        print('特征({}) 的信息增益为： {:.3f}'.format(labels[c], c_info_gain))
    # 比较大小
    best_ = max(best_feature, key=lambda x: x[-1])
    return '特征({})的信息增益最大，选择为根节点特征'.format(labels[best_[0]])

if __name__ == '__main__':
    print(info_gain_train(np.array(datasets)))