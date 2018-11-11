import feedparser
import re
import json 
import os 
import random 
from numpy import *

def loadDataSets():
    if os.path.exists('chicago.json'):
        f = open('chicago.json')
        chicago = json.load(f)
    else:
        chicago0 = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss&s=0')
        chicago1 = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss&s=25')
        chicago2 = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss&s=50')
        chicago3 = feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss&s=75')
        chicago = []
        chicago.extend([item['summary'] for item in chicago0['entries']])
        chicago.extend([item['summary'] for item in chicago1['entries']])
        chicago.extend([item['summary'] for item in chicago2['entries']])
        chicago.extend([item['summary'] for item in chicago3['entries']])
        print len(chicago)
        with open('chicago.json', 'w') as f:
            json.dump(chicago, f)

    if os.path.exists('newyork.json'):
        f = open('newyork.json')
        newyork = json.load(f)
    else:
        newyork0 = feedparser.parse('https://chicago.craigslist.org/search/apa?format=rss&s=0')
        newyork1 = feedparser.parse('https://chicago.craigslist.org/search/apa?format=rss&s=25')
        newyork2 = feedparser.parse('https://chicago.craigslist.org/search/apa?format=rss&s=50')
        newyork3 = feedparser.parse('https://chicago.craigslist.org/search/apa?format=rss&s=75')
        newyork = []
        newyork.extend([item['summary'] for item in newyork0['entries']])
        newyork.extend([item['summary'] for item in newyork1['entries']])
        newyork.extend([item['summary'] for item in newyork2['entries']])
        newyork.extend([item['summary'] for item in newyork3['entries']])
        print len(newyork)
        with open('newyork.json', 'w') as f:
            json.dump(newyork, f)
    
    return chicago, newyork

def textParse(content):
    reg = re.compile('\\W*')
    list = reg.split(content)
    return [tok.lower() for tok in list if len(tok) > 1]
    
def getVocabList(fullText):
    dict = {}
    for word in fullText:
        dict[word] = dict.get(word, 0) + 1
    dictSorted = sorted(dict.iteritems(), key=lambda item:item[1], reverse = True)
    mostFreq = dictSorted[:60]
    for item in mostFreq:
        del dict[item[0]]
    return dict.keys()

def words2Vec(uniqList, item):
    rtnVec = [0] * len(uniqList)
    for word in item:
        if word in uniqList:
            rtnVec[uniqList.index(word)] += 1 
    return rtnVec

def dealWithDataSets():
    ds1, ds2 = loadDataSets()
    minLen = min(len(ds1), len(ds2))
    docList = []
    fullText = []
    classList = []
    for i in range(minLen):
        list = textParse(ds1[i])
        docList.append(list)
        fullText.extend(list)
        classList.append('chicago')

        list = textParse(ds2[i])
        docList.append(list)
        fullText.extend(list)
        classList.append('newyork')
    uniqList = getVocabList(fullText)
    trainSets = []
    for list in docList:
        trainSets.append(words2Vec(uniqList, list))
    return trainSets, classList, uniqList

def train(trainSets, classList):
    numWords = len(trainSets[0])
    numDocChi = 0
    pChi = ones(numWords)
    pNew = ones(numWords)
    numChi = 0
    numNew = 0
    for idx in range(len(trainSets)):
        if classList[idx] == 'chicago':
            numDocChi = numDocChi + 1
            pChi += trainSets[idx]
            numChi += sum(trainSets[idx])
        else:
            pNew += trainSets[idx]
            numNew += sum(trainSets[idx])
    chiVec = log(pChi / numChi)
    newVec = log(pNew / numNew)
    return chiVec, newVec, numDocChi / float(len(trainSets))

def classify(chiVec, newVec, pChi, targetVec):
    chi = sum(chiVec * targetVec) + log(pChi) 
    new = sum(newVec * targetVec) + log(1 - pChi) 
    if chi > new:
        return 'chicago'
    else:
        return 'newyork'

if __name__ == '__main__':
    trainSets, classList, uniqList = dealWithDataSets()
    newTrainSets = []
    newClassList = []
    newTestSets = []
    newTestClassList = []
    for idx in range(len(trainSets)):
        rand = random.random()
        if rand < 0.2:
            newTestSets.append(trainSets[idx])
            newTestClassList.append(classList[idx])
        else:
            newTrainSets.append(trainSets[idx])
            newClassList.append(classList[idx])
            
    chiVec, newVec, pChi = train(newTrainSets, newClassList)
    numFail = 0
    for idx in range(len(newTestSets)):
        predict = classify(chiVec, newVec, pChi, newTestSets[idx])
        if predict != newTestClassList[idx]:
           numFail = numFail + 1
    print numFail
    print len(newTestSets)
    print numFail / float(len(newTestSets)) 

    print '=======chicago========' 
    chiMap = {}
    for i in range(len(chiVec)):
        chiMap[uniqList[i]] = chiVec[i]
    sortedChi = sorted(chiMap.iteritems(), key=lambda s:s[1], reverse = True)
    for i in range(10):
        print sortedChi[i]
        
    print '=======newyork========' 
    newMap = {}
    for i in range(len(newVec)):
        newMap[uniqList[i]] = newVec[i]
    sortedNew = sorted(newMap.iteritems(), key=lambda s:s[1], reverse = True)
    for i in range(10):
        print sortedNew[i]
