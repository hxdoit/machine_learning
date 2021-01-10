# encoding=utf-8
import sys,re
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve,GridSearchCV
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn import metrics
import itertools

pd.set_option('display.expand_frame_repr', False)

def plot_feature_stack_bar(feature):
    print train_data[feature].describe()
    surrived = train_data[feature][train_data.isDefault==1].value_counts()
    unsurrived = train_data[feature][train_data.isDefault==0].value_counts()
    tempView=pd.DataFrame({'yes': surrived, "no": unsurrived})
    #tempView.plot(kind='bar', stacked=True)
    x=[]
    y=[]
    for idx,row in tempView.iterrows():
        #print idx,row
        #print idx,float(row['yes']) / (int(row['yes']) + int(row['no']))
        x.append(int(idx))
        yTemp = 0
        if not row['yes'] and not row['no']:
            yTemp = 0
        elif not row['yes']:
            yTemp = 0
        elif not row['no']:
            yTemp = 1
        else:
            yTemp = float(row['yes']) / (float(row['yes']) + float(row['no']))
        y.append(yTemp)
    print x
    print y
    print '--------'
    print feature
    print train_data[feature].value_counts()
    plt.bar(x,y)
    plt.xlabel(feature)
    plt.show()

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('testA.csv')
test_data_id = test_data.filter(['id'])
#print train_data.head()
#print train_data.info()
#train_data['loanAmnt'] = train_data['loanAmnt'] - train_data['loanAmnt'] % 5000
#yearStr = ['10+ years', '2 years', '< 1 year', '3 years', '1 year', '5 years', '4 years', '8 years', '6 years', '7 years', '9 years']
#yearInt = [10, 2, 0, 3, 1, 5, 4, 8, 6, 7, 9]
def deal(train_data):
    train_data['interestRate'] = train_data['interestRate'] - train_data['interestRate'] % 2
    train_data['installment'] = train_data['installment'] - train_data['installment'] % 100
    #train_data['employmentLength'] = train_data.apply(lambda x: 0 if x['employmentLength'] not in yearStr else yearInt[yearStr.index(x['employmentLength'])], axis=1)
    train_data['annualIncome'] = train_data['annualIncome'] - train_data['annualIncome'] % 5000
    train_data['dti'] = train_data['dti'] - train_data['dti'] % 2

deal(train_data)
deal(test_data)
#plot_feature_stack_bar('n9')

train_label = train_data.filter(['isDefault'])
regex = 'term|interestRate|installment|homeOwnership|annualIncome|delinquency_2years|openAcc|totalAcc|n[1-9]'
attr_list = [column for column in train_data.columns if re.match(regex, column)]
train_data = train_data.filter(attr_list)
test_data = test_data.filter(attr_list)

for column in [column for column in train_data.columns if re.match('n[1-9]', column)]:
    print column
    train_data[column] = train_data.apply(lambda x: 0 if np.isnan(x[column]) else float(x[column]), axis=1)
    test_data[column] = test_data.apply(lambda x: 0 if np.isnan(x[column]) else float(x[column]), axis=1)


X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size=0.05, random_state=0)
def run_model(alg):
    print alg
    alg.fit(X_train, Y_train)

    #print logreg.feature_importances_
    #print logreg.coef_
    #pd.DataFrame(logreg.coef_.ravel(), index=list(train_data.columns)).plot(kind='bar')
    #plt.show()
    #sys.exit(0)

    prepro = alg.predict_proba(X_test)
    print prepro[:, 1]
    print Y_test
    pre=alg.predict(X_test)
    print pre

    acc = alg.score(X_test,Y_test)
    print acc
    #print metrics.accuracy_score(Y_test, pre)
    print metrics.roc_auc_score(Y_test, prepro[:, 1])

    prepro = alg.predict_proba(test_data)
    result = pd.DataFrame({'id':test_data_id['id'].as_matrix(), 'isDefault':prepro[:, 1].astype(np.float)})
    result.to_csv("re.csv", index=False)
    #print np.mean(cross_val_score(LogisticRegression(), X_train, Y_train, cv=5))
#run_model(LogisticRegression(C=1e5))
#run_model(DecisionTreeClassifier())
alg=GradientBoostingClassifier(n_estimators=5, max_features='sqrt', subsample=0.8, random_state=10)
run_model(alg)
#run_model(RandomForestClassifier(random_state=0, n_estimators=2000, n_jobs=-1))


