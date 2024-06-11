from math import log

def createDataSet1():
    datasets = [
        ['是', '是', '是'],
        ['是', '是', '是'],
        ['是', '否', '否'],
        ['否', '是', '否'],
        ['否', '否', '否'],
    ]
    labels = ['不浮出水面是否可以生存', '是否有脚蹼', '属于鱼类']
    return datasets, labels

def calcnum(dataset):
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    return labelCounts

def calcShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = calcnum(dataset)
    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def calccondShannonEnt(dataset, axis):
    data_length = len(dataset)
    feature_sets = {}
    for i in range(data_length):
        feature = dataset[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(dataset[i])
    cond_ShannonEnt = sum([(len(p) / data_length) * calcShannonEnt(p) for p in feature_sets.values()])
    return cond_ShannonEnt

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        infoGain = baseEntropy - calccondShannonEnt(dataSet, axis=i)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda item: item[1], reverse=True)
    return sortedClassCount[0][0]

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

if __name__ == '__main__':
    dataSet, labels = createDataSet1()
    for element in dataSet:
        print(element, end='\n')
    print("labels=", labels)
    count = len(dataSet[0]) - 1
    best_feature = []
    for c in range(count):
        c_ShannonEnt = calccondShannonEnt(dataSet, axis=c)
        info_gain = calcShannonEnt(dataSet) - calccondShannonEnt(dataSet, axis=c)
        best_feature.append((c, info_gain))
        print('初始数据集D的特征({}) 的条件经验熵为： {:.3f}'.format(labels[c], c_ShannonEnt))
        print('初始数据集D的特征({}) 的信息增益为： {:.3f}'.format(labels[c], info_gain))
    best_ = max(best_feature, key=lambda x: x[-1])
    print('特征({})的信息增益最大，选择为根节点特征'.format(labels[best_[0]]))

    tree = createTree(dataSet, labels)
    print("决策树：", tree)
