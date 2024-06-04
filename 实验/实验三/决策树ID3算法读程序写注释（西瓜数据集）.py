from math import log

def createDataSet1():   #这个函数的作用是？
                        ##
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]
    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '类别']
    return dataSet, labels

def calcnum(dataset):  #这个函数的作用是？
                       ##获取好瓜和坏瓜的数量
    labelCounts = {}   #当程序运行完，labelCounts的值为多少？
    for featVec in dataset: #当程序运行完，这个for循环共运行了多少次？featVec指的是什么？
                            ##17，每一个瓜的数据
        currentLabel = featVec[-1]  #当程序第一次运行到这里时，currentLabel等于多少？
                                    ##好瓜
        if currentLabel not in labelCounts.keys():#labelCounts.keys代表什么？
                                                  ##{['好瓜','坏瓜']}
            labelCounts[currentLabel] = 0  #上面if语句和这条语句合在一起实现的作用是什么？
                                           ##如果labelCounts里面没有currentLabel就将currentLabel添加进currentLabels,并将值初始化为0
        labelCounts[currentLabel] += 1  #不管上面的if语句成立与否，这条语句都要执行，对否？
                                        ##对
    return labelCounts

def calcShannonEnt(dataset):#这个函数的作用是？
                            ##求数据集的经验条件熵
    numEntries = len(dataset) #numEntries代表的含义？此时等于多少？
                              ##numEntries代表数据的个数，17
    labelCounts =calcnum(dataset)#这里进行了函数调用，此时labelCounts等于多少？
                                 ##{'好瓜'：8,'坏瓜'：9}
    shannonEnt = 0  #shannonEnt代表的含义是什么？
                    ##经验条件熵
    for key in labelCounts:#key指的是labelCounts的什么？
                           ##标签的数量，好瓜和坏瓜，2
        prob = float(labelCounts[key]) / numEntries#这个for循环中，prob的值分别是多少？
                                                   ##0.47058823...，0.5294117...
        shannonEnt -= prob * log(prob, 2)#这个公式为什么等号前面是个减号？
                                         ##公式为-
    return shannonEnt

def calccondShannonEnt(dataset,axis):#这个函数的作用是？
                                     ##计算条件经验熵
    data_length = len(dataset)# data_length和numEntries是否相等？
                              ##相等
    feature_sets = {}#当axis=0,这个函数运行完后，feature_sets里面的值为多少？
                    ##
    for i in range(data_length):#这个for循环在这里的作用是什么？
        feature = dataset[i][axis]#当程序第一次运行到这里，i=0,axis=0，feature 为多少？
        if feature not in feature_sets:#这里的if语句结合下面的两句代码实现的功能是什么？
            feature_sets[feature] = []
        feature_sets[feature].append(dataset[i])#当axis=0时，上述的for循环遍历完毕后，请问这时的 feature_sets又为多少？
    cond_ShannonEnt = sum([(len(p) / data_length) * calcShannonEnt(p) for p in feature_sets.values()])#这个公式计算的是什么？为什么条件经验熵的公式中没有负号？
    return  cond_ShannonEnt

if __name__ == '__main__':
    dataSet, labels = createDataSet1()#这行代码实现的功能是什么？
                                       ##获取数据集的数据和标题
    for element in dataSet:
        print(element, end='\n') #上述两行代码实现功能是什么？
                                ##将每组打印出来
    print("labels=",labels)
    print("好瓜和坏瓜分别有：",calcnum(dataSet))
    print('初始数据集D的经验熵为： {:.3f}'.format(calcShannonEnt(dataSet)))#上述三行代码实现功能是什么？
                                                                      ##输出数据的标题/好瓜和坏瓜的数量/初始数据集的经验熵
    count = len(dataSet[0]) - 1#count是下面程序代码中c遍历的范围，请问这时c等于多少？回答这个问题，首先回答 len(dataSet[0])等于多少？哪count代表的含义是什么？
                               ##len(dataSet[0])等于7，count代表数据的特征数量
    best_feature = []#  best_feature 代表的是什么，当程序全部运行完， best_feature的值为多少？
                     ##信息增益，值为[(0, 0.10812516526536531), (1, 0.14267495956679288), (2, 0.14078143361499584), (3, 0.3805918973682686), (4, 0.28915878284167895), (5, 0.006046489176565584)]
    for c in range(count):#这个for语句加上后面的5个语句实现的功能是什么？
                          ##
        c_ShannonEnt=calccondShannonEnt(dataSet,axis=c)
        info_gain=calcShannonEnt(dataSet)-calccondShannonEnt(dataSet,axis=c)
        best_feature.append((c,  info_gain))
        print('初始数据集D的特征({}) 的条件经验熵为： {:.3f}'.format(labels[c],  c_ShannonEnt))
        print('初始数据集D的特征({}) 的信息增益为： {:.3f}'.format(labels[c],  info_gain))
    best_ = max(best_feature, key=lambda x: x[-1])  #程序运行到这里best_等于多少？
    print('特征({})的信息增益最大，选择为根节点特征'.format(labels[best_[0]]))
