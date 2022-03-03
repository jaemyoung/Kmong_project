# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:22:42 2022

@author: user
"""

##데이터 setting
#코스피데이터 경로
kospi_file_path = "C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project4/kospi.csv"
#학습데이터 경로
train_file_path ="C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project4/Day_ODS_2022_0219.csv"
#테스트데이터 경로
test_file_path ="C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project4/Day_ODS_2022_0219.csv"
#저장데이터 경로
save_path = "C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project4/result.csv"
##예측값 범위설정
percentile = 0.05


##데이터 전처리
from datetime import datetime, timedelta
import pandas as pd 
#모델 학습데이터 설정
#y_train
kospi = pd.read_csv(kospi_file_path,encoding="cp949",header = 1).dropna()
kospi["날짜"]=pd.to_datetime(kospi["날짜"])#datetime형식으로 변환
y =kospi.set_index(keys=["날짜"])
y_train = y["종가"]
y_train = y_train.reset_index(drop=1)

#X_train
df_train = pd.read_csv(train_file_path)
df_train["date"]=pd.to_datetime(df_train["date"])#datetime형식으로 변환
X = df_train.set_index(keys=["date"])
X_train = X.loc[kospi["날짜"]- timedelta(days =1),:]
X_train = X_train[["Y01-a","Z04-b","Z03-a","Z03-b","Z05-b","Y03-a","X01-b","Y02-a","Z05-a","Y01-b"]].reset_index(drop=1)#변수선택

#X_test
#df_test = pd.read_csv(test_file_path)
#X_test = df_test.drop(["date"],axis=1) #앞으로 예측할 변수파일
df_test = df_train[-6:-1].reset_index(drop=1) # 2/14~ 2/18예측
X_test = df_test.drop(["date"],axis = 1)
##모델링
from lightgbm import LGBMRegressor
LGM_model = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', learning_rate=0.1, max_depth=-1,
              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
              n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
              random_state=10, reg_alpha=0.0, reg_lambda=0.0, silent='warn',
              subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
LGM_model.fit(X_train,y_train)
output = LGM_model.predict(X_test[X_train.columns])
predict = pd.DataFrame(output,columns =["predict"])
top = pd.DataFrame(output*(1+percentile),columns = ["top"],)
bottom = pd.DataFrame(output*(1-percentile),columns = ["bottom"])
result = pd.concat([df_test,bottom,predict,top], axis = 1)

##결과값 저장
result.to_csv(save_path,header =True, index =False)
