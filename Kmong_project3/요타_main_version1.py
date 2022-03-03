# -*- coding: utf-8 -*-
"""#데이터 전처리"""

import pandas as pd
train_path = ("파일경로를 입력")
test_path = ("파일경로를 입력")

train = pd.read_csv(train_path,encoding="UTF-8")
test = pd.read_csv(test_path,encoding="UTF-8")
train = train.drop("sec",axis =1 ) #필요없는 col 제거
test = test.drop("sec",axis =1 ) #필요없는 col 제거

"""변수 정의"""

#Train, Test 분리
y1 = "t1 success"
y2 = "t2 success"
#train set -> 학습시킬 데이터
X_train = train.drop([y1,y2],axis = 1)
y1_train = train[y1]
y2_train = train[y2]
#test set -> 결과를 도출할 데이터
X_test = test.drop([y1,y2],axis = 1)

"""데이터 스케일링""" 
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
#column split
ordinal_col =["N t2 accordance","N t1 accordance","N t2 O(1)/U(0)"]
numeric_col = X_train.columns.drop(ordinal_col)
#스케일링 적용
scaler = StandardScaler()
X_train.loc[:,numeric_col] = scaler.fit_transform(X_train.loc[:,numeric_col])
X_test.loc[:,numeric_col] = scaler.fit_transform(X_test.loc[:,numeric_col])

"""
#t1 success 학습 모델링

학습데이터 정의 및 파라미터 셋팅
"""
from xgboost import XGBClassifier
#t1 trian columns 설정
X_train_t1 = X_train.drop(["diff line"],axis =1 ) # t1 success를 예측할 때 diff line col를 뺀 값이 조금더 성능이 좋음
X_test_t1 = X_test.drop(["diff line"],axis = 1)
#t1 파라미터 설정
estimator_t1 = XGBClassifier(use_label_encoder= False, subsample= 1.0, objective= 'binary:logistic', min_child_weight= 3, max_depth= 4, gamma=1 , colsample_bytree= 0.8)
estimator_t1.fit(X_train_t1,y1_train)
pred_t1 = estimator_t1.predict_proba(X_test_t1)[:,1]


"""
#t2 success 학습 모델링

학습데이터 정의 및 파라미터 셋팅
"""
from xgboost import XGBClassifier
#t2 train columns 설정
X_train_t2 = X_train
X_test_t2 = X_test 
#t2 파라미터 설정
estimator_t2 = XGBClassifier(use_label_encoder= False, subsample= 0.4, objective= 'binary:logistic', min_child_weight= 9, max_depth= 8, gamma=2 , colsample_bytree= 0.6)
estimator_t2.fit(X_train_t2,y2_train)
pred_t2 = estimator_t2.predict_proba(X_test)[:,1]

"""test set에 적용"""

test["t1 success"] = pred_t1
test["t2 success"] = pred_t2
print(test)
