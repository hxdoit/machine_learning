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
    return np.matrix(trainSet)

def plot(trainSet, cent):
    plt.scatter(trainSet[:,0].tolist(), trainSet[:,1].tolist(),s=10, c="#ff1212",marker='o')
    plt.scatter(cent[:,0].tolist(), cent[:,1].tolist(),s=40, c="black",marker='o')
    #plt.scatter(trainSet[:,1].tolist(), predict,s=10, c="blue",marker='o')
    #plt.scatter(trainSet[:,1], trainLabelSet, label="2", color="red", linewidth = 1, linestyle = '-')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.show()

def kmean(dataSet, k):
    m,n=dataSet.shape
    cent = np.matrix(np.zeros((k, n)))
    sta = np.matrix(np.zeros((m, 2)))
    sta[:,0] = -1
    for i in range(n):
        minV = np.min(dataSet[:,i])
        maxV = np.max(dataSet[:,i])
        cent[:,i] = minV + (maxV - minV) * np.random.rand(k, 1)
    hasChanged = True
    while hasChanged:
        print 1
        hasChanged = False
        for i in range(m):
            minIdx = -1
            minDistance = float('inf')
            for j in range(k):
                diff = dataSet[i] - cent[j]
                distance = np.sum(diff * diff.T)
                if distance < minDistance:
                    minIdx = j
                    minDistance = distance
            if sta[i,0] != minIdx:
                hasChanged = True
            sta[i,0] = minIdx
            sta[i,1] = minDistance ** 2
        for centIdx in range(k):
            curCentData = dataSet[(sta[:,0] == centIdx).T.A[0]]
            cent[centIdx] = np.mean(curCentData, axis=0)
    return cent, sta

def bikmeans(dataSet, k):
    m,n=dataSet.shape
    cents = []
    cent,sta = kmean(dataSet, 1)
    cents.append(cent)
    while len(cents) < k:
        minSSE = float('inf')
        #print sta
        for i in range(len(cents)):
            curData = dataSet[(sta[:,0] == i).T.A[0]]
            noSplitSSE = np.sum(sta[(sta[:,0] != i).T.A[0], 1])
            cent1,sta1 = kmean(curData, 2)
            splitSSE = np.sum(cent1[:, 1]) 
            if splitSSE + noSplitSSE < minSSE:
                minSSE = splitSSE + noSplitSSE
                minIdx = i
                minCent = cent1
                minSta = sta1
        minSta[(minSta[:,0]==1).T.A[0], 0] = len(cents)
        minSta[(minSta[:,0]==0).T.A[0], 0] = minIdx
        sta[(sta[:,0] == minIdx).T.A[0]] = minSta
        cents[minIdx]=minCent[0]
        cents.append(minCent[1])
        print minSta
        print cents
    lastCents = np.zeros((k, n))
    for i in range(k):
        lastCents[i] = cents[i]
    return lastCents,sta

if __name__ == '__main__':
    dataSet = getDataSet('testset.txt')
    #dataSet2 = getDataSet('testset2.txt')
    cent, sta = bikmeans(dataSet, 4)
    plot(dataSet, cent)



