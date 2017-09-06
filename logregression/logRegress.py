# coding=utf-8

"""
逻辑回归梯度上升优化算法
"""

from numpy import *


def loadDataSet():
    """
    逐行读取本地文件
    :return:
    """
    dataMat = []
    labelMat = []
    fr = open('files/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """
    实现梯度上升算法
    :param dataMatIn:
    :param classLabels:
    :return:
    """
    dataMatrix = mat(dataMatIn)
    labelMatrix = mat(classLabels).transpose()

    m, n = shape(dataMatrix)

    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMatrix - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度上升，学习率alpha固定大小
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * dataMatrix[i] * error
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    控制迭代次数的随机梯度上升
    迭代过程中动态修改学习率alpha的大小
    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    """
    m, n = shape(dataMatrix)
    weights = ones(n)
    for i in range(numIter):
        dataIndex = range(m)
        for j in range(m):
            alpha = 4 / (1 + i + j) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * dataMatrix[randIndex] * error
    return weights


def plotBestFit(weights):
    """
    画出数据集和Logistic回归最佳拟合直线
    :param weights:
    :return:
    """
    import matplotlib.pyplot as plt

    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]

    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()
