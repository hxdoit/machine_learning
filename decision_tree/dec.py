from math import log

def createDemoDataSet():
    return [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]

def createDataSet():
    fr = open('lenses.txt')
    dataSet = [line.strip().split('\t') for line in fr.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataSet, labels

def calShan(dataSet):
    count = {}
    total = float(len(dataSet)) 
    for line in dataSet:
        label = line[-1]
        count[label] = count.get(label, 0) + 1
    shan = 0.0
    for label in count:
        per = count[label] / total
        shan -= per * log(per, 2)
    return shan 
        
def splitDataSet(dataSet, axis, value):
    rtn = []
    for line in dataSet:
        if line[axis] != value:
            continue
        left = line[:axis]
        right = line[axis + 1:]
        left.extend(right)
        rtn.append(left)
    return rtn

def chooseBestFeature(dataSet):
    feaNum = len(dataSet[0]) - 1
    baseShan = calShan(dataSet)
    infoGain = 0.0
    bestFea = -1
    for i in range(feaNum):
        values = [line[i] for line in dataSet]
        uniqValues = set(values)
        shan = 0.0
        for value in uniqValues:
            subSet = splitDataSet(dataSet, i, value)
            prob = float(len(subSet)) / len(dataSet)
            shan += prob * calShan(subSet)
        tempInfoGain = baseShan - shan
        if tempInfoGain > infoGain:
            infoGain = tempInfoGain
            bestFea = i 
    return bestFea
        
def majority(labels):
    count = {}
    for label in labels:
        count[label] = count.get(label, 0) + 1
    labelSort = sorted(count.iteritems(), key=lambda s:s[1], reverse = True )
    return labelSort[0][0]

def genTree(dataSet, labels):    
    classList = [line[-1] for line in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majority(classList)

    bestFea = chooseBestFeature(dataSet)
    bestFeaName = labels[bestFea]
    tree = {bestFeaName: {}}
    values = [line[bestFea] for line in dataSet]
    uniqValues = set(values)
    for value in uniqValues:
        subSet = splitDataSet(dataSet, bestFea, value)
        tree[bestFeaName][value] = genTree(subSet, labels)
    return tree

def useTree(tree, labels, testVec):
    key = tree.keys()[0]
    detail = tree[key]
    idx = labels.index(key)
     
    sub = detail[testVec[idx]]   
    if type(sub).__name__ == 'dict':
        re = useTree(sub, labels, testVec)
    else:
        re = sub
    return re 
    

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    #dataSet = createDemoDataSet()
    #print calShan(dataSet)
    #print splitDataSet(dataSet, 1, 1)
    tree = genTree(dataSet, labels)
    print useTree(tree, labels, ['young', 'myope', 'no', 'normal'])
