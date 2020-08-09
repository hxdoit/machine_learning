# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
# 加载数据  
def loadDataSet():  
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]  
  
# 创建C1  
def createC1(dataSet):  
    C1 = []  
    for transaction in dataSet:  
        for item in transaction:  
            if not[item] in C1:  
                C1.append([item])  
    C1.sort()  
    return map(frozenset, C1)            #frozenset 可以将集合作为字典的键字使用  
  
# 由Ck生成Lk  
def scanD(D, Ck, minSupport):
    ssCnt = {}  
    for tid in D:  
        for can in Ck:  
            if can.issubset(tid):  
                if not ssCnt.has_key(can):ssCnt[can] = 1  
                else: ssCnt[can] += 1  
    numItems = float(len(D))
    retList = []  
    supportData = {}  
    for key in ssCnt:  
        support = ssCnt[key]/numItems  
        if support >= minSupport:  
            retList.insert(0, key)            #在列表的首部插入任意新的集合  
        supportData[key] = support  
    return retList, supportData  
  
#Apriori 算法  
#  由Lk 产生Ck+1  
def aprioriGen(Lk, k):
    retList = []  
    lenLk = len(Lk)  
    for i in range(lenLk):  
        for j in range(i+1, lenLk):  
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()  
            if L1 == L2:  
                retList.append(Lk[i] | Lk[j])
    return retList  
  
def apriori(dataSet, minSupport = 0.5):  
    C1 = createC1(dataSet)  
    D = map(set, dataSet)  
    L1, supportData = scanD(D, C1, minSupport)  
    L = [L1]  
    k = 2  
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D,Ck, minSupport)  
        supportData.update(supK)  
        L.append(Lk)  
        k += 1  
    return L, supportData  

#从频繁项集中发现关联规则  
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD  
    bigRuleList = []  
    for i in range(1, len(L)):#only get the sets with two or more items  
        for freqSet in L[i]:  
            H1 = [frozenset([item]) for item in freqSet]  
            #print i,H1
            if (i > 1):  
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)  
            else:  
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)  
    return bigRuleList           
  
def calcConf(freqSet, H, supportData, brl, minConf=0.7):  
    prunedH = [] #create new list to return  
    for conseq in H:  
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence  
        if conf >= minConf:   
            print freqSet-conseq,'–>',conseq,'conf:',conf  
            brl.append((freqSet-conseq, conseq, conf))  
            prunedH.append(conseq) 
        else:
            print freqSet-conseq,'not–>',conseq,'conf:',conf  
    return prunedH  
  
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):  
    print freqSet,H
    m = len(H[0])
    if (len(freqSet) > m): #try further merging
        Hmp1 = calcConf(freqSet, H, supportData, brl, minConf) 
        print Hmp1 
        if (len(Hmp1) > 1):    #need at least two sets to merge  
            Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates 
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)  

if __name__ == '__main__':
    dataSet = loadDataSet()
    c1 = createC1(dataSet)
    #print scanD(dataSet,  c1, 0.5)
    #print aprioriGen(c1, 2)
    L,supportData = apriori(dataSet)
    print supportData

    print generateRules(L, supportData, minConf=0.65)

