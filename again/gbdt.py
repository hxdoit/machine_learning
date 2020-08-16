# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import ensemble,metrics
 
 
# 加载sklearn自带的波士顿房价数据集
dataset = load_boston()
print dataset
 
# 提取特征数据和目标数据
X = dataset.data
y = dataset.target
 
# 将数据集以9:1的比例随机分为训练集和测试集，为了重现随机分配设置随机种子，即random_state参数
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=188)
 
# 实例化估计器对象
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gbr = ensemble.GradientBoostingRegressor(**params)
 
# 估计器拟合训练数据
gbr.fit(X_train, y_train)
 
# 训练完的估计器对测试数据进行预测
y_pred = gbr.predict(X_test)
print y_pred
 
# 输出特征重要性列表
print(gbr.feature_importances_)
print(metrics.mean_squared_error(y_test, y_pred))
