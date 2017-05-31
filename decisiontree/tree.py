# coding=utf-8

from math import log
import operator
import treeplotter


def calcShanonEnt(dataSet):
    """
    计算给定数据集的香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    labelCounts = {}

    # 为所有可能类别创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # 以2为底求对数，计算香农熵
    return shannonEnt


def createDataSet():
    """
    创建数据集，
    特征值：不浮出水面是否可以生存；是否有脚蹼
    对应标签：是否是鱼类
    :return:
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    根据给定特征划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分的数据特征
    :param value: 待返回的特征的值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    循环计算香农熵和splitDataSet()，选择用于划分数据集的最好特征
    数据要求：
    1. 所有列表元素具有相同的数据长度
    2. 数据每列最后一个元素表示标签
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShanonEnt(dataSet)
    bestInfoGain = 0.0  # 最好的信息增益
    bestFeature = -1  # 最好特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 创建唯一的分类标签列表
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShanonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i  # 最好特征索引值
    return bestFeature


def majorityCnt(classList):
    """
    数据集处理完了所有属性，但类标签依然不是唯一的
    采用多数表决的方法确定叶子节点分类
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    递归创建树
    两个递归出口：
    1. 所有的类标签完全相同，直接返回该类的标签
    2. 所有特征已经使用完，仍然不能将数据集划分成仅包含唯一类别的分组，返回类标签出现次数最多的分组
    :param dataSet:
    :param labels:
    :return:
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(dataSet):  # 类别完全相同，则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时，返回出现次数最多的
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    使用决策树进行分类
    :param inputTree:
    :param featLabels:
    :param testVec:
    :return:
    """
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) # 将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel



def storeTree(inputTree, filename):
    """
    使用pickle序列化对象并保存
    :param inputTree:
    :param filename:
    :return:
    """
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()


def grabTree(filename):
    """
    使用pickle加载序列化的对象
    :param filename:
    :return:
    """
    import pickle
    fr = open(filename)
    return pickle.load(fr)



if __name__ == '__main__':
    myDat, labels = createDataSet()
    myTree = treeplotter.retrieveTree(0)
    storeTree(myTree, './files/classfifierStorage.txt')
    print grabTree('./files/classfifierStorage.txt')

