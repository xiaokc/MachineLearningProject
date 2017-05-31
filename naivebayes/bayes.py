# coding=utf-8
from numpy import *

"""
使用朴素贝叶斯进行文本分类
这里的特征是来自文本的词条，一个词条是字符的任意组合
"""


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'loved', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 表示侮辱性文字，0表示正常言论
    return postingList, classVec


def createVocabList(dataSet):
    """
    根据文档内容，创建词汇表
    :param dataSet:
    :return:
    """
    vocabList = set([])  # 空表
    for document in dataSet:
        vocabList = vocabList | set(document)  # 两个集合的并集

    return list(vocabList)


def setOfWord2Vec(vocabList, inputSet):
    """
    词向量集合
    使用词集模型：将每个词是否出现作为一个特征，每个词最多出现一次
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            pass
            # print 'the word:{0} is not in my vocabulary!'.format(word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix:
    :param trainCategory:
    :return:
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vec = log(p1Num / p1Denom)
    p0Vec = log(p0Num / p0Denom)

    return p0Vec, p1Vec, pAbusive


def classfyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    根据贝叶斯准则进行计算
    :param vec2Classify: 要分类的向量
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 这里都用的对数运算，ln(a*b)=ln(a)+ln(b)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print testEntry, 'classfied as:', classfyNB(thisDoc, p0V, p1V, pAb)
    testEntry = ['stupid', 'grabage']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print testEntry, 'classfied as:', classfyNB(thisDoc, p0V, p1V, pAb)

