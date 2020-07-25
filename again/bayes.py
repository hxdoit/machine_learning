import feedparser,sys,json,os,re,random,math
import numpy as np

def textParse(str):
    return [item.lower() for item in re.split(r'[^a-zA-Z]', str) if len(item)>3]

def getDataSet():
    sfPath = './sf.json'
    sf = []
    if os.path.exists(sfPath):
        with open(sfPath, 'r') as f:
            sf = json.load(f)
    else:
        rs10 = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss&s=0')
        rs11 = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss&s=25')
        rs12 = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss&s=50')
        rs13 = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss&s=75')
        for i in range(len(rs10)):
            sf.append(rs10['entries'][i]['summary'])
        for i in range(len(rs11)):
            sf.append(rs11['entries'][i]['summary'])
        for i in range(len(rs12)):
            sf.append(rs12['entries'][i]['summary'])
        for i in range(len(rs13)):
            sf.append(rs13['entries'][i]['summary'])
        with open(sfPath, 'w') as f:
            json.dump(sf, f)

    chiPath = './chi.json'
    chi = []
    if os.path.exists(chiPath):
        with open(chiPath, 'r') as f:
            chi = json.load(f)
    else:
        rs10 = feedparser.parse('https://chicago.craigslist.org/search/apa?format=rss&s=0')
        rs11 = feedparser.parse('https://chicago.craigslist.org/search/apa?format=rss&s=25')
        rs12 = feedparser.parse('https://chicago.craigslist.org/search/apa?format=rss&s=50')
        rs13 = feedparser.parse('https://chicago.craigslist.org/search/apa?format=rss&s=75')
        for i in range(len(rs10)):
            chi.append(rs10['entries'][i]['summary'])
        for i in range(len(rs11)):
            chi.append(rs11['entries'][i]['summary'])
        for i in range(len(rs12)):
            chi.append(rs12['entries'][i]['summary'])
        for i in range(len(rs13)):
            chi.append(rs13['entries'][i]['summary'])
        with open(chiPath, 'w') as f:
            json.dump(chi, f)
    
    trainSet=[]
    trainLabelSet=[]
    testSet=[]
    testLabelSet=[]
    for i in range(len(sf)):
        if random.randint(0, len(sf)) < int(0.2*len(sf)):
            testSet.append(textParse(sf[i]))
            testLabelSet.append(1)
        else:
            trainSet.append(textParse(sf[i]))
            trainLabelSet.append(1)
    for i in range(len(chi)):
        if random.randint(0, len(chi)) < int(0.2*len(chi)):
            testSet.append(textParse(chi[i]))
            testLabelSet.append(0)
        else:
            trainSet.append(textParse(chi[i]))
            trainLabelSet.append(0)

    vocList = []
    for i in range(len(trainSet)):
        vocList.extend(trainSet[i])
    for i in range(len(testSet)):
        vocList.extend(testSet[i])
    uniqVocList = getUniqVocList(vocList)
    print len(uniqVocList)
         
    return trainSet, trainLabelSet, testSet, testLabelSet, uniqVocList

def getUniqVocList(fullText):
    dict = {}
    for word in fullText:
        dict[word] = dict.get(word, 0) + 1
    dictSorted = sorted(dict.iteritems(), key=lambda item:item[1], reverse = True)
    mostFreq = dictSorted[:30]
    for item in mostFreq:
        del dict[item[0]]
    return dict.keys()

def text2Bag(line, uniqVocList):
    rtn = np.zeros(len(uniqVocList))
    for i in range(len(line)):
        if line[i] in uniqVocList:
            rtn[uniqVocList.index(line[i])]+=1
    return rtn
        
def train(trainSet, labelSet, uniqVocList):    
    p1Vec=np.ones(len(uniqVocList))
    p0Vec=np.ones(len(uniqVocList))
    for i in range(len(trainSet)):
        tempBag = text2Bag(trainSet[i], uniqVocList)
        if labelSet[i] == 1:
            p1Vec = p1Vec + tempBag
        else:
            p0Vec = p0Vec + tempBag
    return np.log(p1Vec/np.sum(p1Vec)), np.log(p0Vec/np.sum(p0Vec)), float(sum(labelSet))/len(labelSet)

def predict(line, uniqVocList, p1Vec, p0Vec, p1):
    lineBag = text2Bag(line, uniqVocList)
    pp1 = np.sum(p1Vec * lineBag) + math.log(p1)
    pp0 = np.sum(p0Vec * lineBag) + math.log(1 - p1)
    if pp1 > pp0:
        return 1
    else:
        return 0

def getTop20Words(vec, uniqVocList):
    idxSort = np.argsort(-vec)
    for i in range(20):
        print uniqVocList[idxSort[i]]

if __name__ == '__main__':
    trainSet, trainLabelSet, testSet, testLabelSet, uniqVocList = getDataSet()
    p1Vec, p0Vec, p1 = train(trainSet, trainLabelSet, uniqVocList)
    print "------------p1-----------"
    getTop20Words(p1Vec, uniqVocList)
    print "------------p0-----------"
    getTop20Words(p0Vec, uniqVocList)

    succ=0
    for i in range(len(testSet)):
        if predict(testSet[i], uniqVocList, p1Vec, p0Vec, p1) == testLabelSet[i]:
            succ+=1
    print succ, len(testSet), float(succ)/len(testSet)

