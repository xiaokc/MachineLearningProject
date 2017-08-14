# coding=utf-8

import logRegress

if __name__ == '__main__':
    dataMat, labelMat = logRegress.loadDataSet()
    weights = logRegress.gradAscent(dataMat, labelMat)
    print weights
    print logRegress.plotBestFit(weights.getA())
