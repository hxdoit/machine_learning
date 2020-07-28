import numpy as np
from svmutil import *
import os,sys

#https://www.jianshu.com/p/bcc999fc2c8b
#np.set_printoptions(threshold=np.inf)

def img2vec(fileLoc):
    mat = np.zeros(1024)
    f = open(fileLoc, 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip('\n').strip('\r')
        for j in range(len(line)):
            mat[i*32 + j] = int(line[j])
    f.close()
    return mat

def getTrainSet(directory):
    names = os.listdir(directory)
    labels = [] 
    x=[]
    for i in range(len(names)):
        num = names[i].split('_')[0]
        if num == "1":
            labels.append(1.0)
        elif num == "9":
            labels.append(-1.0)
        else:
            continue
        features =  img2vec('%s/%s' % (directory,names[i]))
        dict={}
        for j in range(len(features)):
            dict[j]=features[j]
        x.append(dict)
    return labels,x

if __name__ == '__main__':
    y,x= getTrainSet('../../trainingDigits')
    yTest,xTest= getTrainSet('../../testDigits')
    m = svm_train(y, x, '-c 4') 
    p_label, p_acc, p_val = svm_predict(yTest, xTest, m)
    print p_label,p_acc,p_val
