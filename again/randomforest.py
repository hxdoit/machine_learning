# -*- coding: utf-8 -*-
#导入需要的库  
#https://www.cnblogs.com/zongfa/p/8824351.html
import pandas as pd  
import numpy as np  
from sklearn.ensemble import RandomForestClassifier  
from sklearn import  metrics  
import matplotlib.pylab as plt  

   
#导入数据，顺便看看数据的类别分布  
train= pd.read_csv('train_modified.csv')  
target='Disbursed' # Disbursed的值就是二元分类的输出  
IDcol= 'ID'  
train['Disbursed'].value_counts()  
   
#接着选择好样本特征和类别输出，样本特征为除去ID和输出类别的列  
x_columns = [x for x in train.columns if x not in [target,IDcol]]  
X = train[x_columns]  
y = train['Disbursed']  
   
#不管任何参数，都用默认的，拟合下数据看看  
rf0 = RandomForestClassifier(oob_score=True, random_state=10)  
rf0.fit(X,y)  
print rf0.oob_score_  
y_predprob = rf0.predict_proba(X)[:,1]  
print "AUC Score (Train): %f" % metrics.roc_auc_score(y,y_predprob)  
#输出如下：0.98005  AUC Score (Train): 0.999833  
#可见袋外分数已经很高（理解为袋外数据作为验证集时的准确率，也就是模型的泛化能力），而且AUC分数也很高（AUC是指从一堆样本中随机抽一个，抽到正样本的概率比抽到负样本的概率 大的可能性）。相对于GBDT的默认参数输出，RF的默认参数拟合效果对本例要好一些。 