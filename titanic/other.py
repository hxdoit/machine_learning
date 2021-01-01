# encoding=utf-8
#https://www.jianshu.com/p/ab858fc767b1
#https://www.jianshu.com/p/4c9b49878f3d
#https://www.pianshen.com/article/3462281220/
import matplotlib,sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2


def get_titanic_data():
    train_data =pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    return train_data, test_data

train_data, test_data = get_titanic_data()
#train_data.info()
#train_data.head()


target_column = 'Survived'
continuous_column_list = ['Age', 'SibSp', 'Fare',
                      'Parch']
discrete_column_list = ['Sex', 'Pclass', 'Embarked']
text_column_list = ['Name', 'Ticket', 'Cabin']

continuous_train_data = train_data.filter(continuous_column_list)
discrete_train_data = train_data.filter(discrete_column_list)

print "Parch:", chi2(train_data.filter(["Parch"]), train_data['Survived'])
print "SibSp:", chi2(train_data.filter(["SibSp"]), train_data['Survived'])

feature = 'Parch'
feature_data = train_data.filter([feature, 'Survived'])
survived_data = feature_data[feature][feature_data.Survived == 1].value_counts()
unsurvived_data = feature_data[feature][feature_data.Survived == 0].value_counts()
df = pd.DataFrame({'Survived': survived_data, 'UnSurvivied': unsurvived_data})
df.plot(kind='bar', stacked=True)
plt.title('Survived_' + feature)
plt.xlabel(feature)
plt.ylabel(u'Number of people')
#plt.show()

feature_data.groupby(feature).hist()

for column in discrete_train_data.columns:
    print discrete_train_data[column].value_counts()

from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
sex_label_data = LabelBinarizer().fit_transform(train_data['Sex'])
embarked_label_data = LabelEncoder().fit_transform(train_data['Embarked'].fillna('S'))
print "Embarked", chi2(pd.DataFrame(embarked_label_data), train_data['Survived'])
print "Sex:", chi2(sex_label_data, train_data['Survived'])
print "Pclass:", chi2(train_data.filter(["Pclass"]), train_data['Survived'])


def print_stacked_hist(feature):
    feature_data = train_data.filter([feature, 'Survived'])
    survived_data = feature_data[feature][feature_data.Survived == 1].value_counts()
    unsurvived_data = feature_data[feature][feature_data.Survived == 0].value_counts()
    df = pd.DataFrame({'Survived': survived_data, 'UnSurvivied': unsurvived_data})
    df.plot(kind='bar', stacked=True)
    plt.title('Survived_' + feature)
    plt.xlabel(feature)
    plt.ylabel(u'Number of people')
    plt.show()

#print_stacked_hist('Sex')
#print_stacked_hist('Pclass')
#print_stacked_hist('Embarked')


filled_data = train_data.copy()
# transform Age
filled_data.loc[np.isnan(train_data['Age']), 'Age'] = 0
def transform_category(data, start, step, category):
    """
    data是一个array数据
    """
    result = ((data - start) / step).astype(int) + category
    return result

step = 5
filled_data['Age'] = transform_category(filled_data['Age'], 0, step, 0)

filled_data.loc[filled_data['Cabin'].notnull(), 'Cabin'] = 1
filled_data.loc[filled_data['Cabin'].isnull(), 'Cabin'] = 0

def get_most_common_category(series):
    #print series.value_counts()
    return series.value_counts().axes[0][0]

most_common = get_most_common_category(filled_data['Embarked'])
filled_data.loc[filled_data['Embarked'].isnull(), 'Embarked'] = most_common
#print filled_data

dummy_cabin = pd.get_dummies(filled_data["Cabin"], prefix="Cabin")
dummy_sex = pd.get_dummies(filled_data['Sex'], prefix='Sex')
dummy_embarked = pd.get_dummies(filled_data['Embarked'], prefix='Embarked')

dummied_data = pd.concat([filled_data, dummy_cabin, dummy_sex, dummy_embarked], axis=1)
dummied_data.drop(['Cabin', 'Sex', 'Embarked'], axis=1, inplace=True)
#print dummied_data.head()

from sklearn.preprocessing import StandardScaler
dummied_data['Fare'] = StandardScaler().fit_transform(dummied_data.filter(['Fare']))
#print dummied_data.head()

unsed_column = ['PassengerId', 'Name', 'Ticket']
target_prepared_y = dummied_data['Survived']
train_prepared_data = dummied_data.drop(unsed_column + ['Survived'], axis=1)





from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn import metrics
def modelfit(alg, X, Y, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    #alg.fit(X, Y)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, X, Y, cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(Y.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(Y, dtrain_predprob))
    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
def train_model(model_class, print_coef=False, *args, **kwargs):
    kf = KFold(n_splits=10)
    best_lr = None
    best_score = 0
    for train_index, test_index in kf.split(train_prepared_data):
        train_sub_data, target_sub_data = train_prepared_data.loc[train_index], target_prepared_y.loc[train_index]
        test_sub_data, test_target_sub_data = train_prepared_data.loc[test_index], target_prepared_y.loc[test_index]
        lr = model_class(*args, **kwargs)
        lr.fit(train_sub_data, target_sub_data)
        score = lr.score(test_sub_data, test_target_sub_data)
        if score > best_score:
            best_lr = lr
            best_score = score

    print best_score
    print best_lr

    modelfit(best_lr, train_prepared_data, target_prepared_y, printFeatureImportance=False)
    if print_coef:
        columns = list(train_prepared_data.columns)
        plot_df = pd.DataFrame(best_lr.coef_.ravel(), index=columns)
        plot_df.plot(kind='bar')
        #plt.show()
    return best_lr
train_model(LogisticRegression, print_coef=True)

from sklearn.tree import DecisionTreeClassifier
model = train_model(DecisionTreeClassifier, max_depth=5)

from IPython.display import Image,display
from sklearn import tree
import pydotplus
#Image( filename =  '/Users/huangxuan01/Downloads/a.jpg' )
dot_data = tree.export_graphviz(model, out_file=None, 
                         feature_names=list(train_prepared_data.columns), 
                         class_names=['UnSurvived', 'Survived'],
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
#print type(graph)
pic=graph.create_png()
f = open('test.png','w')
f.write(pic)
f.close()

child_age = 15 / 5
old_man_age = 50 / 5

(train_data['Fare'] > 80).value_counts()
rich_fare = 1
def add_new_features(df):
    df['Child'] = df['Age'].apply(lambda x: 1 if 0 < x<= child_age else 0)
    df['Senior'] = df['Age'].apply(lambda x: 1 if x >= old_man_age else 0)
    df['Rich'] = df['Fare'].apply(lambda x: 1 if x >= rich_fare else 0)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Single'] = df['Parch'].apply(lambda x: 1 if x == 0 else 0)
    return

add_new_features(train_prepared_data)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
saved_list = ['Sex', 'Pclass']
def choose_feature_list(X, y, saved_list):
    left_columns = list(X.columns)
    lr = LogisticRegression()
    best_score = np.mean(cross_val_score(lr, X, y, cv=5))
    final_column_list = [x for x in left_columns]
    while len(left_columns) >= 3:
        new_item = None
        for item in left_columns:
            if item in saved_list:
                continue
            
            lr = LogisticRegression()
            new_column = [x for x in left_columns]
            new_column.remove(item)
            score = np.mean(cross_val_score(lr, X.filter(new_column), y, cv=5))
            if score > best_score:
                best_score = score
                new_item = item
        
        if new_item is None:
            break
        
        final_column_list.remove(item)
        print "remove:", new_item, "best_score ever", best_score
        left_columns.remove(item)

choose_feature_list(train_prepared_data, target_prepared_y, saved_list)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import re
import itertools
drop_str = 'Age|SibSp|Parch|Cabin_*|Embarked_*|Senior|Rich|Single'
drop_list = [column for column in train_prepared_data.columns if re.match(drop_str, column)]
print drop_list

def choose_feature_list(X, y, drop_list):
    
    lr = LogisticRegression()
    best_score = np.mean(cross_val_score(lr, X, y, cv=5))
    final_drop_list = []
    
    cnt = 0
    for i in xrange(1, len(drop_list) + 1):
        drop = None
        for combination in itertools.combinations(drop_list, i):
            cnt += 1
            new_train_X = X.drop(list(combination), axis=1)
            
            lr = LogisticRegression()
            score = np.mean(cross_val_score(lr, new_train_X, y, cv=5))
            
            if score > best_score:
                best_score = score
                final_drop_list = combination
                drop = True
            
            if cnt % 100 == 0:
                print cnt
        if drop:
            print "drop_list:", final_drop_list, "best_score ever", best_score

#choose_feature_list(train_prepared_data, target_prepared_y, drop_list)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"sample n")
        plt.ylabel(u"score")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"cv")

        plt.legend(loc="best")
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

#plot_learning_curve(LogisticRegression(), u"learning curve", train_prepared_data, target_prepared_y);


from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV   #Perforing grid search
clf = LogisticRegression()
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8)
bagging_clf.fit(train_prepared_data, target_prepared_y)
print bagging_clf.score(train_prepared_data, target_prepared_y)
#drop_data = dummied_prepard_data.drop(remove_feature_list, axis=1)
#result = model.predict(drop_data)
#result = model.predict(drop_data)
#passage_list = test_data['PassengerId']
# print_result(passage_list, result)


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier()
#new_prepared = train_prepared_data.drop(remove_feature_list, axis=1)
model = train_model(GradientBoostingClassifier)



param_test = {
#     'n_estimators': range(20, 10, 40),
    'min_samples_split':range(2, 20, 4), 
    'min_samples_leaf':range(1,10,2), 
    'learning_rate': (0.05, 0.2, 0.05), 
    'max_depth': range(6, 10, 2)
}
gv_search = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=5, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test, scoring='accuracy', iid=False, cv=5)
gv_search.fit(train_prepared_data, target_prepared_y)
print gv_search.best_score_
#drop_data = dummied_prepard_data.drop(remove_feature_list, axis=1)
#result = model.predict(drop_data)
#passage_list = test_data['PassengerId']
#print_result(passage_list, result)


