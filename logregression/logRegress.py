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
