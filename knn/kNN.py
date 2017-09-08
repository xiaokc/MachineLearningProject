# coding=utf-8
from numpy import *
import operator

"""
K 临近算法，工作原理：
样本集中每个数据存在标签，输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，
然后算法提取样本集中前K个特征最相似的（最邻近）的分类标签，选择K个最相似数据中出现次数最多的分类，
作为新数据的分类。
"""


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    KNN分类简单实现：
    1.计算距离 2.选择距离最小的k个点  3.排序
    :param inX: 待分类的输入向量
    :param dataSet: 输入的样本集
    :param labels: 标签向量
    :param k: 选择最近邻居的个数
    :return:
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distancess = sqDistances ** 0.5
    sortedDistIndices = distancess.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    将文本记录转换到Numpy
    :param filename: 约会数据文本
    :return:
    """
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)  # 得到文件行数

    returnMat = zeros((numberOfLines, 3))  # 创建返回的Numpy矩阵
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    归一化特征值
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape(0)
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    分类器针对约会网站的测试代码
    :return:
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "分类结果:%d，真实标签：%d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):  # 记录分类错误的个数
            errorCount += 1

    print '分类错误率:%f' % (errorCount / float(numTestVecs))


def classifyPerson():
    """
    约会网站预测函数
    :return:
    """
    resultLsit = ['不太喜欢', '一般喜欢', '非常喜欢']

    percentTats = float(raw_input("每天打游戏时长"))
    exerciseTats = float(raw_input("每周运动时长"))
    dirtyWords = float(raw_input("每周脏话数"))

    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([percentTats, exerciseTats, dirtyWords])  # 输入待预测向量
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print "你对该约会对象的感觉可能是:", resultLsit[classifierResult - 1]


if __name__ == '__main__':
    group, labels = createDataSet()
    print classify0([0, 0], group, labels, 3)
