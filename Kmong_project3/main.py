# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 19:35:19 2022

@author: user
"""

import pandas as pd
import re
##데이터 불러오기
data_path = ("C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project3/sample+data.csv")
data = pd.read_csv(data_path,encoding="UTF-8")
data = data.drop("sec",axis =1 )

#이상치 제거
outlier = data[data["TOTAL A diff"] == -1987.639692].index
data = data.drop(outlier, axis =0)
data.describe()

##변수 정의
#Train, Test 분리
y1 = "t1 success"
y2 = "t2 success"
train = data[:-6]
test = data[-6:]

#train set
X_train = train.drop([y1,y2],axis = 1)
y1_train = train[y1]
y2_train = train[y2]
#test set
X_test = test.drop([y1,y2],axis = 1)
y1_test = test[y1]
y2_test = test[y2]

#column split
ordinal_col =["N t2 accordance","N t1 accordance","N t2 O(1)/U(0)"]
numeric_col = X_train.columns.drop(ordinal_col)

    
#상관분석(Correlation)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(40,20))
sns.heatmap(train.corr(), annot=True)
plt.title("Correlation")

#t1 success에 영향을 주는 변수들
t1 = abs(train.corr()["t1 success"]).drop("t1 success").sort_values()
x = np.arange(len(t1))
plt.figure(figsize=(20,15))
plt.bar(x,t1.values)
plt.xticks(x,t1.index)

#t2 success에 영향을 주는 변수들
t2 = abs(train.corr()["t2 success"]).drop("t2 success").sort_values()
x = np.arange(len(t2))
plt.figure(figsize=(20,15))
plt.bar(x,t2.values)
plt.xticks(x,t2.index)
plt.show()

# 설명 변수들의 분포를 확인하며 정규성을 띄는지 확인 
dfX = pd.DataFrame(X_train.drop(ordinal_col,axis =1))
dfy = pd.DataFrame(y1_train)
feature_names = dfX.columns

for i in range(dfX.shape[1]):
    sns.distplot(dfX.iloc[i])
    plt.title(feature_names[i])
    plt.show()


#데이터 스케일링
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler

#스케일링 적용
scaler = MaxAbsScaler()
X_train.loc[:,numeric_col] = scaler.fit_transform(X_train.loc[:,numeric_col])
X_test.loc[:,numeric_col] = scaler.fit_transform(X_test.loc[:,numeric_col])

## t1 success 학습 모델링

from sklearn.model_selection import train_test_split
X_train_split, X_valid, y1_train_split, y1_valid = train_test_split(X_train, y1_train, random_state=1, test_size = 0.2)

#모델 학습
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV

#모델 및 변수 정의
model = XGBClassifier()
X_model = X_train_split
y_model = y1_train_split
X_test = X_valid
y_test = y1_valid

#파라미터 seting
xgb_parameters = {
 'min_child_weight': [0.5, 1, 3, 5, 8],
 'gamma': [0.2, 0.5, 1, 2],
 'subsample': [0.4, 0.6, 0.8, 1.0],
 'colsample_bytree': [0.2, 0.4, 0.6, 0.8],
 'max_depth': [4, 6, 8, 10],
 'objective': ['binary:logistic'],
 'use_label_encoder': [False] 
}

cv = KFold(n_splits=6)

rsv = RandomizedSearchCV(model, xgb_parameters, cv=cv, scoring='roc_auc', n_jobs=6, verbose=10)
rsv.fit(X_model.values,y_model.values)

print('final params', rsv.best_params_)
print('best score', rsv.best_score_)

estimator = rsv.best_estimator_
estimator.fit(X_model,y_model)
pred_xgb = estimator.predict(X_test)

#모델 평가
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

roc_score = roc_auc_score(y_test, pred_xgb)
print('t1 success ROC AUC 값 : {0:.4f}'.format(roc_score))

#모델의 변수별 중요도 계산
xgboost.plot_importance(estimator)
plt.show()

#confusion matrix
cf_matrix = confusion_matrix(y_test, pred_xgb)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt=".1f")
ax.set_title('t1 success Confusion Matrix with labels\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

plt.show()


## t2 success 학습 모델링

from sklearn.model_selection import train_test_split
X_train_split, X_valid, y2_train_split, y2_valid = train_test_split(X_train,y2_train, random_state=1, test_size = 0.2)

#모델 학습
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV

#모델 및 변수 정의
model = XGBClassifier()
X_model = X_train_split
y_model = y2_train_split
X_test = X_valid
y_test = y2_valid

#파라미터 seting
xgb_parameters = {
 'min_child_weight': [0.5, 1, 3, 5, 8],
 'gamma': [0.2, 0.5, 1, 2],
 'subsample': [0.4, 0.6, 0.8, 1.0],
 'colsample_bytree': [0.2, 0.4, 0.6, 0.8],
 'max_depth': [4, 6, 8, 10],
 'objective': ['binary:logistic'],
 'use_label_encoder': [False] 
}

cv = KFold(n_splits=6)

rsv = RandomizedSearchCV(model, xgb_parameters, cv=cv, scoring='roc_auc', n_jobs=6, verbose=10)
rsv.fit(X_model.values,y_model.values)

print('final params', rsv.best_params_)
print('best score', rsv.best_score_)

estimator = rsv.best_estimator_
#estimator = model
estimator.fit(X_model,y_model)
pred_xgb = estimator.predict(X_test)

#모델 평가
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

roc_score = roc_auc_score(y_test, pred_xgb)
print('t2 success ROC AUC 값 : {0:.4f}'.format(roc_score))

#모델의 변수별 중요도 계산
xgboost.plot_importance(estimator)
plt.show()

#confusion matrix
cf_matrix = confusion_matrix(y_test, pred_xgb)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt=".1f")
ax.set_title('t2 success Confusion Matrix with labels\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

plt.show()






