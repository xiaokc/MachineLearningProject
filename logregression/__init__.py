# coding=utf-8

import logRegress

if __name__ == '__main__':
    dataMat, labelMat = logRegress.loadDataSet()
    print logRegress.gradAscent(dataMat, labelMat)
