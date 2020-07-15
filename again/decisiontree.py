import numpy as np
import os,sys,math

#np.set_printoptions(threshold=np.inf)

def createDataSet():
    f = open('lenses.txt')
    lines = f.readlines()
    dataset = []
    for i in range(len(lines)):
        dataset.append(lines[i].strip('\n').strip('\r').split("\t"))
    return dataset


def createTestDataSet():
    return [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]

def calcEntropy(dataset):
    statistics = {}
    for i in range(len(dataset)):
        label = dataset[i][-1]
        statistics[label] = statistics.get(label, 0) + 1
    entropy = 0
    for key in statistics.keys():
        prop = float(statistics.get(key)) / len(dataset)
        entropy -= math.log(prop, 2) * prop
    return entropy

def splitDataSet(dataset, axis, value):
    rtn = []
    for i in range(len(dataset)):
        if dataset[i][axis] == value:
            line = dataset[i][:axis]
            line.extend(dataset[i][axis+1:])
            rtn.append(line)
    return rtn

def chooseBestFeature(dataset):
    baseEnt = calcEntropy(dataset)
    bestFea = -1
    infoGain = 0
    numFeatures = len(dataset[0]) - 1
    for i in range(numFeatures):
        values = [x[i] for x in dataset]
        distinctVal = set(values)
        featureInfo = 0
        for uniqVal in distinctVal:
            subDataSet = splitDataSet(dataset, i, uniqVal)
            subEnt = calcEntropy(subDataSet)
            featureInfo += float(len(subDataSet)) * subEnt / len(dataset)
        tempInfoGain = baseEnt - featureInfo
        if tempInfoGain > infoGain:
            infoGain = tempInfoGain
            bestFea = i
    return bestFea

def genTree(dataset, labels):
    uniqLabel = set([line[-1] for line in dataset])
    if len(uniqLabel) == 1:
        return uniqLabel.pop()
    numFeatures = len(dataset[0]) - 1
    if numFeatures < 1:
        majority = {}
        for i in range(len(dataset)):
            majority[dataset[i][-1]] = majority.get(dataset[i][-1], 0) + 1
        sort = sorted(majority.items(), key=lambda x:x[1], reverse=True)
        return sort[0][0]

    feaIdx = chooseBestFeature(dataset)
    tree = {labels[feaIdx]: {}}

    values = [x[feaIdx] for x in dataset]
    distinctVal = set(values)
    subLabels = labels[:]
    del(subLabels[feaIdx])
    for value in distinctVal:
        tree[tree.keys()[0]][value] = genTree(splitDataSet(dataset, feaIdx, value), subLabels)
    return tree

if __name__ == '__main__':
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    print genTree(createDataSet(), labels)
        
    
