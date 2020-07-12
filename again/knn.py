import numpy as np
import os,sys

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
    features = np.zeros((len(names), 32 * 32)) 
    for i in range(len(names)):
        labels.append(names[i].split('_')[0])
        features[i,:] =  img2vec('%s/%s' % (directory,names[i]))
    return labels,features
        
def classify(testFeature, trainFeatures, trainLabels, k): 
    trainNum = trainFeatures.shape[0]
    distance=np.zeros(trainNum)
    for i in range(trainNum):
        distance[i] = np.sum((trainFeatures[i] - testFeature)**2)
    topkIdx = np.argsort(distance)[0:k]
    rtn={}
    for i in range(k): 
        topkNum = trainLabels[topkIdx[i]]
        if not rtn.has_key(topkNum):
            rtn[topkNum] = 0
        rtn[topkNum] = rtn[topkNum] + 1
    sort=sorted(rtn.items(),key=lambda x:x[1],  reverse=True)
    return sort[0][0]

if __name__ == '__main__':
    #print img2vec('trainingDigits/9_31.txt') 
    labels,trains = getTrainSet('trainingDigits')
    names =  os.listdir('testDigits')
    succ=0
    fail=0
    for i in range(len(names)):
        label = names[i].split('_')[0]
        feature =  img2vec('%s/%s' % ('testDigits',names[i]))
        if classify(feature, trains, labels, 3) == label:
            succ = succ+1
        else:
            fail = fail + 1
    print succ
    print fail 
    print float(succ)/(succ+fail)
        
    
