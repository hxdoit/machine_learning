import numpy  as np
import matplotlib.pyplot as plt
import re
def loadDataSet(fileName):
    dataSet = []; labelSet = []
    fr =  open(fileName)
    for line in fr.readlines():
        lineArr = re.split("\s*", line)
        dataSet.append([float(lineArr[0]), float(lineArr[1])])
        labelSet.append(float(lineArr[2]))
    return dataSet, labelSet

def showDataSet(dataMat, labelMat):
    data_plus = []                                 
    data_minus = []                                
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)             
    data_minus_np = np.array(data_minus)           
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])  
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()


if __name__ == '__main__':
    dataSet, labelSet = loadDataSet('dataset')
    showDataSet(dataSet, labelSet) 
