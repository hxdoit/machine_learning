import numpy as np
import matplotlib.pyplot as plt
import math

def getDataSet(file):
    trainSet = []
    f = open(file, 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        lineArr = lines[i].replace('\n', '').split('\t')
        lineArr = map(float, lineArr)
        trainSet.append(lineArr)
    #return np.matrix(trainSet)[:,[0, 1]].astype(float), np.matrix(trainSet)[:,2].astype(float)
    return np.matrix(trainSet)

def splitDataSet(dataSet, idx,  value):
    choose = np.array((dataSet[:,idx]>value).T)[0]
    reverse = (1 - choose).astype(bool)
    return dataSet[choose], dataSet[reverse]

def leafType1(dataSet): 
    return np.mean(dataSet[:, -1])

def errorType1(dataSet):
    return np.var(dataSet[:, -1]) * len(dataSet)


def regLeafType(dataSet):
    m,n=dataSet.shape
    x = dataSet[:, :n-1]
    y = dataSet[:,-1]
    if np.linalg.det(x.T * x) == 0.0:
        raise Exception('mat.I not exists')
    w = (x.T * x).I * x.T * y
    return w

def regErrorType(dataSet):
    m,n=dataSet.shape
    x = dataSet[:, :n-1]
    y = dataSet[:,-1]
    w = regLeafType(dataSet)
    diff = x * w - y
    return np.sum(diff.T * diff)

def chooseBestFeature(dataSet, leafType, errorType, ops=(1, 4)):
    print len(dataSet)
    tolS = ops[0]
    tolN = ops[1]
    m, n = dataSet.shape
    curError = errorType(dataSet)
    errorMin = float('inf')
    featureIdx = -1
    value = -1
    for i in range(n - 1):
        for j in set(dataSet[:, i].T.tolist()[0]):
            left, right = splitDataSet(dataSet, i, j)
            if len(left) < tolN or len(right) < tolN:
                continue
            newError = errorType(left) + errorType(right)
            if newError < errorMin:
                errorMin = newError
                featureIdx = i
                value = j
    #print curError
    #print errorMin
    #print curError - errorMin
    if abs(curError - errorMin) < tolS:
        return None, leafType(dataSet)
    left1, right1 = splitDataSet(dataSet, featureIdx, value)
    if len(left1) < tolN or len(right1) < tolN:
        return None, leafType(dataSet)
    return featureIdx, value


def createTree(dataSet, leafType, errorType, ops):
    tree ={}
    bestFeature, value = chooseBestFeature(dataSet, leafType, errorType, ops)
    if bestFeature == None:
        tree['value'] = value
        return tree
    left, right = splitDataSet(dataSet, bestFeature, value)
    tree['bestFeature'] = bestFeature
    tree['value'] = value
    tree['left'] = createTree(left, leafType, errorType, ops)
    tree['right'] = createTree(right, leafType, errorType, ops)
    return tree

def isTree(tree):
    return tree.has_key('bestFeature')

def getMean(tree):
    if not isTree(tree):
        return tree['value']
    if isTree(tree['left']):
        tree['left']['value'] = getMean(tree['left'])
        del tree['left']['bestFeature'] 
        del tree['left']['left'] 
        del tree['left']['right'] 
    if isTree(tree['right']):
        tree['right']['value'] = getMean(tree['right'])
        del tree['right']['bestFeature'] 
        del tree['right']['left'] 
        del tree['right']['right'] 
    return (tree['left']['value'] + tree['right']['value']) / 2

def prune(tree, testData, errorType):
    if testData.shape[0] == 0:
        return {'value': getMean(tree)}
    if isTree(tree):
        left, right = splitDataSet(testData, tree['bestFeature'], tree['value'])
        leftTree = prune(tree['left'], left, errorType)
        rightTree = prune(tree['right'], right,  errorType)
        if not isTree(leftTree) and not isTree(rightTree):
            splitError = np.sum(np.power(left[:, -1] - leftTree['value'], 2)) + np.sum(np.power(right[:, -1] - rightTree['value'], 2))
            treeMean = (leftTree['value'] + rightTree['value']) / 2
            noSplitError = np.sum(np.power(testData[:, -1] - treeMean, 2))
            if splitError > noSplitError:
                tree['value'] = treeMean
                del tree['bestFeature']
                del tree['left']
                del tree['right']
                print "merge"
                return tree
            else:
                tree['left'] = leftTree
                tree['right'] = rightTree
                return tree
        else:
            tree['left'] = leftTree
            tree['right'] = rightTree
            return tree
    return tree

def pruneBook(tree, testData):
    if testData.shape[0] == 0:
        return {'value': getMean(tree)}
    if isTree(tree['left']) or isTree(tree['right']):
        left,right=splitDataSet(testData, tree['bestFeature'], tree['value'])
    if isTree(tree['left']):
        leftTree = pruneBook(tree['left'], left)
        tree['left'] = leftTree
    if isTree(tree['right']):
        rightTree = pruneBook(tree['right'], right)
        tree['right'] = rightTree

    if not isTree(tree['left']) and not isTree(tree['right']):
        left,right=splitDataSet(testData, tree['bestFeature'], tree['value'])
        splitError = np.sum(np.power(left[:, -1] - tree['left']['value'], 2)) + np.sum(np.power(right[:, -1] - tree['right']['value'], 2))
        treeMean = (tree['left']['value'] + tree['right']['value']) / 2
        noSplitError = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if splitError > noSplitError:
            tree['value'] = treeMean
            del tree['bestFeature']
            del tree['left']
            del tree['right']
            print "merge"
            return tree
        else:
            return tree
    else:
        return tree

def mkdata():
    x = []
    for i in range(20):
        x.append([1.0, i, 0.1*i + 3])
    for i in range(21, 100):
        x.append([1.0, i, 3*i + 2])
    return np.matrix(x)


def plot(trainSet):
    plt.scatter(trainSet[:,1    ].tolist(), trainSet[:,-1].tolist(),s=10, c="#ff1212",marker='o')
    #plt.scatter(trainSet[:,1].tolist(), predict,s=10, c="blue",marker='o')
    #plt.scatter(trainSet[:,1], trainLabelSet, label="2", color="red", linewidth = 1, linestyle = '-')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.show()

if __name__ == '__main__':
    #trainSet = getDataSet('ex2train.txt')
    #trainSet = getDataSet('ex000.txt')
    trainSet = mkdata()
    #print trainSet
    #plot(trainSet)
    #testSet = getDataSet('ex2.txt')
    #plot(trainSet)
    #print splitDataSet(np.matrix(np.eye(4)), 1, -0.5)
    #tree = createTree(trainSet, leafType, errorType, (0, 1))
    print createTree(trainSet, regLeafType, regErrorType, (10, 3))
    #print tree
    #print pruneBook(tree, testSet)
    #print prune(tree, testSet, errorType)
