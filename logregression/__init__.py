# coding=utf-8

import logRegress
from numpy import *

if __name__ == '__main__':
    dataMat, labelMat = logRegress.loadDataSet()
    weights = logRegress.stocGradAscent1(array(dataMat), labelMat)
    print weights
    print logRegress.plotBestFit(weights)
