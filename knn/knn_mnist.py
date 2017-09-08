# coding=utf-8

from numpy import *
from os import listdir

from kNN import classify0


def img2vector(filename):
    """
    将32*32的二维图像矩阵转换成1*1024的向量
    :param filenamed:
    :return:
    """
    returnVec = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readlines()
        for j in range(32):
            returnVec[0, 32 * i + j] = int(lineStr[j])
    return returnVec


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # 获取目录文件
    m = len(trainingFileList)
    trainingMat = zeros(m, 1024)

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])  # 从文件名解析分类数字
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')  # 获取测试目录文件
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print '分类结果是：%d, 真是结果是：%d' % (classifierResult, classNumStr)

        if (classifierResult != classNumStr):
            errorCount += 1.0
    print '分类错误的总数是：%d' % errorCount
    print '总的错误率：%f' % (errorCount / float(mTest))
