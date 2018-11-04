from numpy import *
import os, sys, re

def img2vector(fileName):
    fd = open(fileName, 'r')
    content = fd.read()
    contentList = zeros(1024) 
    idx = 0
    for i in range(len(content)):
        if content[i] != '0' and content[i] != '1':
            continue
        contentList[idx]= int(content[i])
        idx = idx + 1
    return contentList

def createDataSet():
    rootDir = './digits/trainingDigits';
    list = os.listdir(rootDir)
    group = zeros((len(list), 1024))
    labels = []
    files = []
    for fileIdx in range(len(list)):
        fileName = os.path.join(rootDir, list[fileIdx])
        group[fileIdx, :] = img2vector(fileName) 
        num = list[fileIdx].split('_')[0]
        labels.append(num)
        files.append(list[fileIdx])
    return group, labels, files

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
    dataSet, labels, files = createDataSet()
    rootDir = './digits/testDigits';
    list = os.listdir(rootDir)
    right = 0
    wrong = 0
    for fileIdx in range(len(list)):
        fileName = os.path.join(rootDir, list[fileIdx])
        testFileVector = img2vector(fileName) 
        trueNum = list[fileIdx].split('_')[0]
        predictNum = classify0(testFileVector, dataSet, labels, 5) 
        if trueNum != predictNum:
            print list[fileIdx], predictNum
            wrong = wrong + 1
        else:
            right = right + 1
    print "%d,%d,%.3f" % (wrong, right, float(wrong)/(right + wrong))

