# encoding=utf-8
#https://www.cnblogs.com/nolonely/p/6955696.html
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import sys
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.feature_selection import SelectKBest,f_classif 
from sklearn.ensemble import GradientBoostingClassifier   
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score

data_train=pd.read_csv("train.csv")
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
titanic = set_Cabin_type(data_train)

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

#https://www.sohu.com/a/194347159_752099
#data_train['Sex_type'] =  data_train.Sex.map({'male':0,'female':1})
#features = ['Sex_type', 'Age', 'Pclass']
#x=data_train.loc[:,features]
#y=data_train.Survived
#dtc = DecisionTreeClassifier()
#scores = cross_val_score(dtc,x,y,cv=10,scoring='accuracy')
#print scores.mean()
#data_train.fillna(data_train.Age.mean(), inplace=True)
#print data_train
#sys.exit(0)


titanic = pd.concat([titanic, dummies_Sex], axis=1)

predictors = ["Age","SibSp","Parch","Fare","Sex1","Sex2","Pclass"]  

#Perform feature selection  
selector=SelectKBest(f_classif,k=5)  
selector.fit(titanic.filter(regex='Age|SibSp|Parch|Fare|Sex_.*|Pclass'),titanic["Survived"])  
  
#Plot the raw p-values for each feature,and transform from p-values into scores  
scores=-np.log10(selector.pvalues_)  
  
#Plot the scores.   See how "Pclass","Sex","Title",and "Fare" are the best?  
plt.bar(range(len(predictors)), scores)  
plt.xticks(range(len(predictors)), predictors,rotation='vertical')  
#plt.show()


algorithms=[  \
    [GradientBoostingClassifier(random_state=1,n_estimators=25,max_depth=3), ["Pclass","Sex","Age","Fare","Age","Embarked"]],  \
    [linear_model.LogisticRegression(random_state=1),["Pclass","Sex","Fare","Age","Embarked"]] \
]  

titanic_surrived=titanic["Survived"]
titanic = titanic.filter(regex='Age|SibSp|Parch|Fare|Sex_.*|Pclass')
#Initialize the cross validation folds  
kf=KFold(n_splits=3,random_state=1).split(titanic)
  
predictions=[]  
for train,test in kf:  
    train_target=titanic_surrived.iloc[train]  
    full_test_predictions=[]  
    #Make predictions for each algorithm on each fold  
    for alg,predictors in algorithms:  
        #Fit the algorithm on the training data  
        alg.fit(titanic.iloc[train,:],train_target)  
        #Select and predict on the test fold  
        #The .astype(float) is necessary to convert the dataframe to all floats and sklearn error.  
        temp=alg.predict_proba(titanic.iloc[test,:].astype(float))[:,1]  
        full_test_predictions.append(temp)
    #Use a simple ensembling scheme -- just  average the predictions to get the final classification.  
    test_predictions=(full_test_predictions[0]+full_test_predictions[1])/2  
    #Any value over .5 is assumed to be a 1 prediction,and below .5 is a 0 prediction.  
    test_predictions[test_predictions<=0.5]=0  
    test_predictions[test_predictions>0.5]=1  
    predictions.append(test_predictions)  
  
#Put all the predictions together into one array.  
predictions=np.concatenate(predictions,axis=0)  
  
#Compute accuracy by comparing to the training data  
accuracy=sum(predictions[predictions==titanic_surrived])/len(predictions)  
print(accuracy)   
