import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt    #导入matplotlib库
import matplotlib as mpl
import operator                    #导入operator库
import numpy as np


def createDataSet():               #创建数据集
    group=np.array([[64,104],[57,100],[34,81],[18,52],[40,63],[55,24],[87,34],[101,33],[99,21],[54,8]])
    labels=['喜剧片','喜剧片','喜剧片','喜剧片','喜剧片','动作片','动作片','动作片','动作片','动作片','动作片']
    return group,labels

def draw(group,labels):           #定义draw函数，画样本散点
    mpl.rcParams['font.family'] = ['sans-serif']
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False     #设置显示字体
    for i in range(len(group)):   #for循环遍历所有样本点
        if labels[i]=='喜剧片':    #if判断样本点的标签为喜剧片，样本点画位绿色圆点，group[i][0]为样本点横坐标，group[i][1]样本点纵坐标
           plt.scatter(group[i][0], group[i][1], marker='o', color='green', s=50)
        else:                    #if判断样本点的标签为动作片，样本点画位红色方框，group[i][0]为样本点横坐标，group[i][1]样本点纵坐标
            plt.scatter(group[i][0], group[i][1], marker='s', color='red', s=50)
    plt.xlabel("打斗镜头")       #画散点图横坐标
    plt.ylabel("笑点镜头")       #画散点图纵坐标
    plt.show()

def clcEuclideanDis(test,group):   #定义clcEuclideanDis函数，功能是计算测试点与已知样本点之间的欧式距离
    dataSetSize =group.shape[0]    #计算样本点个数，dataSetSize=10
    diffMat = np.tile(test, (dataSetSize, 1)) - group.reshape(10, 2)  #np.tile()函数的作用是将test数组沿第一个轴重复10次，第二轴重复1次，这样就创建了一个更大的数组，才方便与group数组进行相减（同维）
    sqDiffMat = diffMat ** 2 #减得的结果进行平方
    sqDistance = sqDiffMat.sum(axis=1)  #平方求和,对每行进行求和，参数为1
    distance = sqDistance ** 0.5        #求和以后再开根号
    return distance

def classify(distance,k):         #定义classify函数，功能是对上述欧式距离进行排序，并根据K值，确定离测试点最近K个点的类别，并返回最多类别
    sortedDisIndices = distance.argsort()     #对上述函数求得的欧式距离distance进行排序，注意argsort（）的用法,argsort()函数是NumPy库中的一个函数,用于返回数组元素排序后的索引值。
    print(sortedDisIndices)
    classCount = {}                           #设置一个空字典classCount
    for i in range(k):                        #通过for循环，找到距离最近的K个点
        voteIlabel = labels[sortedDisIndices[i]]  #找到前K个点对应的labels
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #对前K个点的labels进行计数，注意get()函数的使用
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #并对得到的classCount进行排序，注意sorted()函数的使用
    print(classCount)
    return sortedClassCount[0][0]

if __name__=="__main__":
    group,labels=createDataSet()   #调用createDataSet函数，得到group和labels
    print(group)                   #打印group
    print(labels)                  #打印labels
    a = int(input("打斗镜头"))      #输入测试电影的打斗镜头
    b = int(input("笑点镜头"))      #输入测试电影的笑点镜头
    test = [a, b]                 #测试电影
    plt.scatter(a, b, marker='*', color='black', s=80) #在散点图上画出测试电影对应的坐标，黑色星号
    draw(group, labels)
    Distance=clcEuclideanDis(test, group)     #调用clcEuclideanDis函数，计算测试电影与已知样本点的欧式距离，返回距离数组
    print("Distance=",Distance)               #打印距离
    k=int(input("输入k近邻的k值"))              #人工输入K值
    test_class = classify(Distance,3)        #通过K值和欧式距离确定测试样本属于哪类电影
    print("测试电影类型为:" + test_class)       #输出测试电影类型
