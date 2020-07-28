import numpy as np
import math, random
#http://archive.ics.uci.edu/ml/datasets/Horse+Colic

def getDataSet(file):
    trainSet = []
    trainLabelSet = []
    f = open(file, 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        line = []
        lineArr = lines[i].split(' ')
        for j in range(22):
            if j == 2:
                continue
            line.append(0 if lineArr[j] == '?' else float(lineArr[j]))
        label = 0 if lineArr[22] == '?' else float(lineArr[22])
        if label != 0:
            trainSet.append(np.array(line))
            trainLabelSet.append(1.0 if label == 1.0 else 0.0)
    return trainSet, trainLabelSet

def sigmod(x):
    return 1.0 / (1.0 + math.exp(-1.0 * x))

def train(dataSet, labelSet):
    n = len(dataSet[0])
    m = len(dataSet)
    w = np.ones(n)
    iter = 5000
    for i in range(iter):
        copy = range(m)
        for j in range(m):
            randIdx = random.randint(0, len(copy) - 1)
            alpha = 0.01
            h = sigmod(np.sum(w * dataSet[copy[randIdx]]))
            error = labelSet[copy[randIdx]] - h
            w = w + alpha * error * dataSet[copy[randIdx]]
            del copy[randIdx]
    return w


if __name__ == '__main__':
     trainSet, trainLabelSet = getDataSet('horse-colic.data')
     testSet, testLabelSet = getDataSet('horse-colic.test')
     w = train(trainSet, trainLabelSet)
     print w
     succ = 0
     for i in range(len(testSet)):
        predict = 1.0 if sigmod(np.sum(w * testSet[i])) > 0.5 else 0.0
        if predict == testLabelSet[i]:
            succ+=1
     print succ, len(testSet), float(succ) / len(testSet)




