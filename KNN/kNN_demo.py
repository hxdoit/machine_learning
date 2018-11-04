from numpy import *

def createDataSet():
    group = array([[1.0, 1.1],  [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,  labels

def classify0(inX, dataSet, labels, k):
    shape = dataSet.shape
    formatInX = tile(inX, (shape[0], 1))
    diff = (formatInX - dataSet)**2
    sum = diff.sum(axis=1)
    distance = sum**0.5
    sortIdx = argsort(distance)
    count = {}
    i = 0
    while i < k:
        count[labels[sortIdx[i]]] = count.get(labels[sortIdx[i]], 0) + 1
        i = i + 1 
    re = sorted(count.iteritems(), key = lambda s:s[1], reverse = True)
    return re[0][0]

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print classify0([0.7, 0.1], dataSet,  labels, 2) 
