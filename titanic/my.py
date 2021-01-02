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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import itertools

train_data = pd.read_csv('train.csv')
#print train_data.head()
#print train_data.info()

def plot_feature_stack_bar(feature):
    surrived = train_data[feature][train_data.Survived==1].value_counts()
    unsurrived = train_data[feature][train_data.Survived==0].value_counts()
    pd.DataFrame({'surrived': surrived, "unsurrived": unsurrived}).plot(kind='bar', stacked=True)
    plt.xlabel(feature)
    plt.show()

for column in train_data.columns:
    print column
    #print train_data[column].value_counts()
    #plot_feature_stack_bar(column)

train_data['Age'] = train_data.apply(lambda x: 0 if np.isnan(x['Age']) else int(x['Age'] / 5), axis=1)
train_data['Sex'] = train_data.apply(lambda x: 0 if x['Sex'] == 'male' else 1, axis=1)

for column in ['Age', 'Pclass', 'Sex', 'Parch', 'SibSp']:
    print column
    #print chi2(train_data.filter([column]), train_data.filter(['Survived']))

#sys.exit(0)

dummy_pclass = pd.get_dummies(train_data['Pclass'], prefix='Pclass')
dummy_embarked = pd.get_dummies(train_data['Embarked'], prefix='Embarked')
#print type(dummy_embarked)

#print type(train_data.filter(['Age']))
std = StandardScaler()
train_data['Age'] = std.fit_transform(train_data.filter(['Age']))
train_data['Parch'] = std.fit_transform(train_data.filter(['Parch']))
train_data['SibSp'] = std.fit_transform(train_data.filter(['SibSp']))

train_data = pd.concat([train_data, dummy_embarked, dummy_pclass], axis=1)
train_label = train_data.filter(['Survived'])
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Pclass', 'Embarked', 'Survived'], axis=1, inplace=True)

regex = 'Sex|Age|SibSp|Parch|Embarked_.*|Pclass_.*'
drop_list = [column for column in train_data.columns if re.match(regex, column)]
def choose_feature_list(x, y, drop_list):
    best_score = np.mean(cross_val_score(LogisticRegression(), x, y, cv=5))
    print best_score
    cnt=0
    cur_drop=[]
    #print drop_list, best_score
    for i in range(1, len(drop_list)):
        drop=False
        for j in itertools.combinations(drop_list, i):
            cnt = cnt+1
            new_train_data = x.drop(list(j), axis=1)
            score = np.mean(cross_val_score(LogisticRegression(), new_train_data, y, cv=5))
            if score > best_score:
                drop=True
                best_score = score
                cur_drop = j
            if cnt % 100 == 0:
                print cnt
        if drop:
            print best_score,cur_drop
    return cur_drop

#need_drop = choose_feature_list(train_data, train_label, drop_list)

X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size=0.3, random_state=0)
def run_model(alg):
    print alg
    alg.fit(X_train, Y_train)

    #print logreg.feature_importances_
    #print logreg.coef_
    #pd.DataFrame(logreg.coef_.ravel(), index=list(train_data.columns)).plot(kind='bar')
    #plt.show()
    #sys.exit(0)

    prepro = alg.predict_proba(X_test)
    pre=alg.predict(X_test)

    acc = alg.score(X_test,Y_test)
    print acc
    print metrics.accuracy_score(Y_test, pre)
    print metrics.roc_auc_score(Y_test, prepro[:, 1])
    #print np.mean(cross_val_score(LogisticRegression(), X_train, Y_train, cv=5))

run_model(LogisticRegression(C=1e5))
run_model(DecisionTreeClassifier())
run_model(GradientBoostingClassifier(n_estimators=5, max_features='sqrt', subsample=0.8, random_state=10))

def leaning_curve():
    train_size, train_score, test_score = learning_curve(LogisticRegression(), train_data, train_label, cv=5, train_sizes=np.linspace(0.05, 1.0, 20))
    train_score_mean = np.mean(train_score, axis=1)
    test_score_mean = np.mean(test_score, axis=1)
    plt.plot(train_size, train_score_mean)
    plt.plot(train_size, test_score_mean)
    plt.show()


def grid():
    gridH=GridSearchCV(LogisticRegression(), param_grid={'C': range(2, 100, 5)}, cv=5)
    gridH.fit(train_data, train_label)
    print gridH.best_score_
    print gridH.best_params_
    print gridH.best_estimator_

grid()


