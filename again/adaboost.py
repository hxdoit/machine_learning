import numpy as np
import math

def loadSimpleData():
    dataMat = np.matrix([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.0],
        [1.0, 1.0],
        [2.0, 1.0]
        ])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

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
            trainSet.append(line)
            trainLabelSet.append(1.0 if label == 1.0 else -1.0)
    return np.matrix(trainSet), trainLabelSet

def judge(dataSet, featureIdx, compare, value):
    m, n = dataSet.shape
    rtn=[0.0]*m
    matrixRtn=[]
    if compare == 'lt':
        matrixRtn = dataSet[:,featureIdx] <= value
    else:
        matrixRtn = dataSet[:,featureIdx] > value
    for i in range(m):
        rtn[i] = -1.0 if matrixRtn[i][0] else 1.0
    return rtn

def oneLevelDs(dataSet, classLabels, d):
    m, n = dataSet.shape
    wError=1e10
    featureIdx = -1
    value = -1
    compare = ''
    predict = []
    for i in range(n):
        max = np.amax(dataSet[:,i])
        min = np.amin(dataSet[:, i])
        steps = 10
        inc = (max - min) / steps
        for j in range(-1, steps+1):
            cur = min + j * inc
            compares = ['lt', 'gt']
            for k in compares:
                rtn1 = judge(dataSet, i, k, cur)
                errorNew = np.sum((np.array(rtn1)!=np.array(classLabels)).astype(int)*np.array(d))
                if errorNew < wError:
                    wError = errorNew
                    featureIdx = i
                    value = cur
                    compare = k
                    predict = rtn1
                #print i, cur, k, errorNew, predict
    return wError, featureIdx, value, compare, predict

def train(dataSet, classLabels):
    d = [1.0 / dataSet.shape[0]] * dataSet.shape[0]
    addedPredict = np.zeros(dataSet.shape[0])
    trees = []
    iter = 50
    for i in range(iter):
        tree={}
        wError, featureIdx, value, compare, predict = oneLevelDs(dataSet, classLabels, d)
        print i, "wError, featureIdx, value, compare, predict:", wError, featureIdx, value, compare, predict
        tree['wError'] = wError
        tree['featureIdx'] = featureIdx
        tree['value'] = value
        tree['compare'] = compare
        alpha = 0.5 * math.log((1 - wError) / (1e-16 if wError == 0 else wError))
        tree['alpha'] = alpha
        trees.append(tree)

        print "alpha:",alpha
        addedPredict += alpha * np.array(predict)
        print "addedPredict:", addedPredict
        totalFail = np.sum((np.sign(addedPredict) != np.array(classLabels)).astype(int))
        print "totalFail:",totalFail
        if totalFail == 0:
            break

        for j in range(len(classLabels)):
            d[j] = math.exp(-alpha) * d[j] if predict[j] == classLabels[j] else math.exp(alpha) * d[j]
        d = (np.array(d) / sum(d)).tolist()
        print "D:", d
        print '---------\n'

    return trees

def classify(line, trees):
    addedPredict = 0
    for i in range(len(trees)):
        rtn = judge(line, trees[i]['featureIdx'], trees[i]['compare'], trees[i]['value'])
        addedPredict += trees[i]['alpha'] * rtn[0]
    return np.sign(addedPredict)

if __name__ == '__main__':
    #dataSet, classLabels = loadSimpleData()
    trainSet, trainLabelSet = getDataSet('horse-colic.data')
    testSet, testLabelSet = getDataSet('horse-colic.test')
    trees = train(trainSet, trainLabelSet)
    succ = 1.0
    for i in range(len(testSet)):
        predict = classify(testSet[i], trees)
        if predict == testLabelSet[i]:
            succ += 1
    print succ, len(testSet), succ/len(testSet)




