import numpy as np
import matplotlib.pyplot as plt
import math

def getDataSet(file):
    trainSet = []
    f = open(file, 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        lineArr = lines[i].replace('\n', '').split('\t')
        trainSet.append(lineArr)
    return np.matrix(trainSet)[:,[0, 1]].astype(float), np.matrix(trainSet)[:,2].astype(float)

def trainLWLR(trainSet, trainLabelSet):
    predict = []
    m = len(trainSet)
    k = 0.003
    for i in range(m):
        w = np.zeros((m, m))
        for j in range(m):
            distance = math.pow(((trainSet[i] - trainSet[j]) * (trainSet[i] - trainSet[j]).T)[0][0], 1)
            w[j][j] = math.exp(distance / (-2.0 * k ** 2))
        if np.linalg.det(trainSet.T * w * trainSet) == 0.0:
            raise Exception('mat.I not exists')
        wPredict = (trainSet.T * w * trainSet).I * trainSet.T * w * trainLabelSet
        predict.append((trainSet[i,[0, 1]]*wPredict)[0][0])
    return predict

def train(trainSet, trainLabelSet):
    if np.linalg.det(trainSet.T * trainSet) == 0.0:
        raise Exception('mat.I not exists')
    w = (trainSet.T * trainSet).I * trainSet.T * trainLabelSet
    return w

def trainRidge(trainSet, trainLabelSet):
    #trainLabelSetCopy = trainLabelSet - np.mean(trainLabelSet, axis=0)
    trainLabelSetCopy = trainLabelSet
    lam = math.exp(-1)
    mean = np.mean(trainSet, axis=0)
    var = np.var(trainSet, axis=0)
    for i in range(var.shape[1]):
        var[0, i] = var[0, i] if var[0, i] != 0.0 else 1.0
    #trainSetCopy = (trainSet - mean) / var
    trainSetCopy = trainSet
    demon = trainSetCopy.T * trainSetCopy + np.eye(trainSet.shape[1]) * lam
    if np.linalg.det(demon) == 0.0:
        raise Exception('mat.I not exists')
    w = (demon).I * trainSetCopy.T * trainLabelSetCopy
    return w

def trainStage(trainSet, trainLabelSet, eps = 0.001, iter = 10000):
    m,n = trainSet.shape
    mean = np.mean(trainSet, axis=0)
    var = np.var(trainSet, axis=0)
    for i in range(var.shape[1]):
        var[0, i] = var[0, i] if var[0, i] != 0.0 else 1.0
    trainSetCopy = (trainSet - mean) / var
    w = np.zeros((n, 1))
    for i in range(iter):
        print w
        wMax = w.copy()
        error = float('inf')
        for j in range(n):
            for k in [-1, 1]:
                wTest = w.copy()
                wTest[j][0] += eps * k
                tempError = (trainSet * wTest - trainLabelSet).T * (trainSet * wTest - trainLabelSet)
                if tempError < error:
                    error = tempError
                    wMax = wTest
        w = wMax
    return w



def plot(trainSet, trainLabelSet, w):
    plt.scatter(trainSet[:,1].tolist(), trainLabelSet.tolist(),s=10, c="#ff1212",marker='o')
    plt.scatter(trainSet[:,1].tolist(), (trainSet[:,[0, 1]]*w).tolist(),s=10, c="blue",marker='o')
    print np.corrcoef((trainSet[:,[0, 1]]*w).T, trainLabelSet.T)
    #plt.scatter(trainSet[:,1], trainLabelSet, label="2", color="red", linewidth = 1, linestyle = '-')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.show()

def plotLWLR(trainSet, trainLabelSet, predict):
    plt.scatter(trainSet[:,1].tolist(), trainLabelSet.tolist(),s=10, c="#ff1212",marker='o')
    plt.scatter(trainSet[:,1].tolist(), predict,s=10, c="blue",marker='o')
    #plt.scatter(trainSet[:,1], trainLabelSet, label="2", color="red", linewidth = 1, linestyle = '-')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC")
    plt.show()

if __name__ == '__main__':
    trainSet, trainLabelSet = getDataSet('ex0.txt')
    #w = train(trainSet, trainLabelSet)
    #plot(trainSet, trainLabelSet, w)
    #predict = trainLWLR(trainSet, trainLabelSet)
    #plotLWLR(trainSet, trainLabelSet, predict)
    #w = trainRidge(trainSet, trainLabelSet)
    #print w
    w = trainStage(trainSet, trainLabelSet)
    plot(trainSet, trainLabelSet, w)




