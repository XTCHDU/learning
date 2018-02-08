# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 11:35:38 2018

@author: Administrator
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import RandomizedLasso,LinearRegression,Lasso,LassoLars,LassoCV
from sklearn.grid_search import GridSearchCV  
from sklearn.svm import SVR  
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import LeaveOneOut
import mlbox
d1 = pd.read_excel('训练.xlsx')
d2 = pd.read_excel('测试B.xlsx')
d3 = pd.read_excel('测试A.xlsx')
d4 = pd.read_csv('answer_a.csv',header=None)
d3['Value']=d4[1]
d1 = d1.drop(303)
#d1 = d1.sort_values(by='Y') 
d1_d2 = pd.concat([d1,d3,d2],axis=0).reset_index()
del d1_d2['index']
del d1_d2['ID']
Isduplicated = d1_d2.T.duplicated()
d1_d2 = d1_d2.iloc[:, [i for i in range(d1_d2.shape[1]) if not Isduplicated[i]]]


###去重
from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *
paths = ["./data/b榜训练.csv", "./data/b榜测试.csv","./data/b榜验证.csv"] #to modify
target_name = "Value" #to modify
data = Reader(sep=",").train_test_split(paths, target_name)  #reading
data = Drift_thresholder().fit_transform(data)  #deleting non-stable variables
space = {
        'ne__numerical_strategy' : {"space" : ['median', 'mean']},

        'ce__strategy' : {"space" : ["label_encoding"]},

        'fs__strategy' : {"space" : ["variance", "rf_feature_importance",'l1']},
        'fs__threshold': {"search" : "uni", "space" : [ 0,0.5]},


        'est__num_leaves' : {"search" : "choice", "space" : [7,12]},
        'est__subsample' : {"search" : "uniform", "space" : [0.8,0.9]},
        'est__reg_lambda' : {"search" : "uniform", "space" : [0,0.2]},


        }
best=Optimiser(scoring="mean_squared_error",n_folds=10).optimise(space,data,20)
Predictor().fit_predict(best,data)
o = ODPS('LTAIl0YqtSU8j0YT', 'pXBkTszTfgjKpNGBhK0mct5vzzyarq', 'tianchi_gongyeyuce',
         endpoint='http://service.odps.aliyun.com/api'
            )

###预处理 分离哑变量 提取日期 去除方差为0列

dum = pd.DataFrame()
drop = []
date = pd.DataFrame()
for col in d1_d2.columns:
    if d1_d2[col].dtype == 'object' :
        temp = pd.get_dummies(d1_d2[col],prefix=col)
        dum = pd.concat([dum,temp],axis=1)
        del d1_d2[col]
    else:
        if  str(d1_d2[col].median()).startswith('2017') or str(d1_d2[col].median()).startswith('2016'):
            date = pd.concat([date,d1_d2[col]],axis=1)
            del d1_d2[col]
        else:
            if d1_d2[col].var()==0.0:
                del d1_d2[col]
d1_d2 = d1_d2.dropna(how='any',axis=1,thresh =1000)
d1_d2 = pd.concat([d1_d2,dum],axis=1)


###时间特征处理
def jian(t,s1,s2):
    
    ans = pd.DataFrame(t[s1]-t[s2])
    for i,row in ans.iterrows():
        if np.isnan(t.ix[i,s1]) or np.isnan(t.ix[i,s2]):
            row[0]==np.nan
    ans.columns = [s1+'_'+s2]
    return ans
def get_p (t,d1):
    ans = []
    scaler = MinMaxScaler(feature_range=(1,5))
    for col in t.columns:
        coef_1,p = pearsonr(t[col],d1.Y)
        coef_2,p = pearsonr(pd.DataFrame(np.log(scaler.fit_transform(t[col].reshape(-1,1)))).ix[:,0],d1.Y)
        coef_3,p = pearsonr(t[col]**2,d1.Y)
        
        ans.append([abs(coef_1),1,col])
        ans.append([abs(coef_2),2,col])
        ans.append([abs(coef_3),3,col])
    
    return pd.DataFrame(ans)
def search(t,d1):
    ans = pd.DataFrame()
    for col1 in range(t.shape[1]-1):
        for col2 in range(col1+1,t.shape[1]):
            temp = jian(t,t.columns[col2],t.columns[col1])
            t1 = get_p(temp,d1)
            ans = pd.concat([ans,t1],axis=0)
    return ans
def get_date(t):
    ans = pd.DataFrame()
    for col in range(t.shape[1]-1):
        for col2 in range(col+1,t.shape[1]):
            if len(str(t.ix[0,col])) == len(str(t.ix[0,col2])):
                new = pd.DataFrame(t.ix[:,col]-t.ix[:,col2])
                new.columns=[t.columns[col]+'_'+t.columns[col2]]
                ans = pd.concat([ans,new],axis=1)
                
    return ans
new_date = get_date(date)
#pear_date = pd.DataFrame(search(date.fillna(value = date.median()).head(500),d1))

def find_yc(t,head,tail):
    ans = []
    for col in t.columns:
        t1,q1 =pearsonr(t.ix[:head,col],t.ix[:head,'Y'])
        t2,q2 =pearsonr(t.ix[tail:,col],t.ix[tail:,'Y'])
        t3,q3 =pearsonr(t[col],t.Y)
        
        ans.append([col,abs(t1),abs(t2),abs(t3)])
    return ans
def shaixuan(t,th):
   ans = pd.DataFrame()
   for i,row in t.iterrows():
       if row[1]>th or row[2]>th or row[3]>th:
           ans = pd.concat([ans,pd.DataFrame(row).T],axis=0)
   return ans
def pick(t):
    drop = []
    count=0
    for i,row in t.iterrows():
        for j in range(count):
            if abs(row[j])>0.99:
                drop.append(i)
                break
        count+=1
    return drop
new_date['Y']=d1_d2.Value
yc_date = pd.DataFrame(find_yc(new_date.fillna(new_date.median()).head(1099),100,1000))
yc_date = yc_date.ix[(yc_date.ix[:,1:3].drop_duplicates()).index,:]
sx_yc_date = shaixuan(yc_date,0.1)
f_date = new_date.ix[:,sx_yc_date.ix[:,0]]
drop_date = pick(f_date.corr())
for col in drop_date:
    del f_date[col]
###温度特征处理
#m = d1_d2.median()
#wd = d1_d2[m[(m>100) & (m<2000) & (d1_d2.std()<500)].index]
#wd['Y']=d1_d2.Y
#yc_wd = pd.DataFrame(find_yc(wd.fillna(wd.median()).head(500),40,460))
#sx_yc_wd = shaixuan(yc_wd,0.3)
#f_wd = wd.ix[:,sx_yc_wd.ix[:,0]]

###温度差特征


def get_wd(t):
    ans = pd.DataFrame()
    for col in range(t.shape[1]-1):
        print(col)
        for col2 in range(col+1,t.shape[1]):
            new = pd.DataFrame(t.ix[:,col]-t.ix[:,col2])
            new.columns=[t.columns[col]+'_'+t.columns[col2]]
            ans = pd.concat([ans,new],axis=1)
            
    return ans
m = d1_d2.median()
wd = d1_d2[m[(m>600) & (m<1400) ].index]

wd_delta = get_wd(wd)
wd_delta['Y']=d1_d2.Value
yc_wd_delta = pd.DataFrame(find_yc(wd_delta.fillna(wd_delta.median()).head(1099),100,1000))
sx_yc_wd_delta = shaixuan(yc_wd_delta,0.1)
f_wd_delta = wd_delta.ix[:,sx_yc_wd_delta.ix[:,0]]
drop_wd = pick(f_wd_delta.corr())
for col in drop_wd:
    del f_wd_delta[col]

###总特征

yc_d1_d2 = pd.DataFrame(find_yc(d1_d2.fillna(d1_d2.median()).head(499),100,400))
sx_yc_d1_d2 = shaixuan(yc_d1_d2,0.1)
f_d1_d2 = d1_d2.ix[:,sx_yc_d1_d2.ix[:,0]]
drop_d1_d2 = pick(f_d1_d2.corr())
for col in drop_d1_d2:
    del f_d1_d2[col]
del f_date['Y']
del f_wd_delta['Y']
del f_d1_d2['Y']

###PCA and 划分数据集
ans = pd.read_excel('答案A.xlsx',header=None)
f_data = pd.concat([f_date,f_wd_delta,f_d1_d2],axis=1)
f_data.fillna(value=f_data.mean(),inplace=True)
scaler = MinMaxScaler()
f_data = scaler.fit_transform(f_data)
pca=PCA(0.87)
f_data = pca.fit_transform(f_data)
yy = d1_d2.Y
f_data = pd.concat([pd.DataFrame(f_data),yy],axis=1)

n=[]
for C in n:
    model = SVR(C) 
    model.fit(f_data.ix[:498,:-1],f_data.ix[:498,-1])
    print(mean_squared_error(model.predict(f_data.ix[620:,:-1]),ans[1]))
    




f_data = pd.concat([f_date,f_wd_delta,d1_d2],axis=1)
f_data.fillna(value=f_data.median(),inplace=True)
scaler = MinMaxScaler()
f_data = scaler.fit_transform(f_data)
pca=PCA(0.87)
f_data = pca.fit_transform(f_data)
f_data = pd.concat([pd.DataFrame(f_data),yy],axis=1)
def my_sample(t,test_size):
    train = pd.DataFrame()
    test = pd.DataFrame()
    m = 500//test_size
    for i,row in t.iterrows():
        if i % m == 0:
            test = pd.concat([test,row.T],axis=1)
        else:
            train = pd.concat([train,row.T],axis=1)
    return train,test
train,test = my_sample(f_data.head(499),100)
train=train.T
test = test.T
train_y = train.Y
test_y = test.Y
del train['Y']
del test['Y']

###开始训练
def get_weight(t):
    ans = np.zeros(t.shape[0])
    count=0
    for i in range(t.shape[0]):
        if (t[i]<2.7 or t[i]>3):
            ans[count]=3
        else:
            ans[count]=1
        count+=1
    return ans
                
def model_test(data,model):

    ans = []
    for test_size in [10,30,50,100]:
        train,test = my_sample(data.head(499),test_size)
        train=train.T
        test = test.T
        train_y = train.Y
        test_y = test.Y
        del train['Y']
        del test['Y']
        model.fit(train,train_y,sample_weight=(train_y-2.85)**2)
        ans.append(mean_squared_error(model.predict(test),test_y))
    return ans,sum(ans)/4
    
    
n=[]
for C in n:
    print(model_test(f_data.ix[:498,:-1],f_data.ix[:498,-1],SVR(C=C)))    
    
    
n = [1]
for C in n:
    print(learning_curve(estimator=Lasso(alpha=C), X = f_data.ix[:498,:-1],y = f_data.ix[:498,-1], groups=None, 
                   train_sizes=np.array([ 1. ]),
                   cv=KFold(3), scoring='neg_mean_squared_error', exploit_incremental_learning=False, n_jobs=1,
                   pre_dispatch='all', verbose=0, shuffle=False, random_state=np.random.randint(2017))[2].mean())
n=[1]
for C in n:
    model_xgb = Lasso(alpha=1e-2)
    model_xgb.fit(f_data[:499],yy.head(499),sample_weight=(train_y-2.85)**2)
    print(mean_squared_error(model_xgb.predict(f_data[620:]),ans[1]))




f_data = pd.concat([f_date,f_wd_delta,d1_d2],axis=1)
f_data.fillna(method='bfill',inplace=True)
scaler = MinMaxScaler()
f_data = scaler.fit_transform(f_data)
pca=PCA(0.87)
f_data = pca.fit_transform(f_data)
f_data = pd.concat([pd.DataFrame(f_data),yy],axis=1)

model = GradientBoostingRegressor(max_depth=3,n_estimators=100,criterion='mse'
        ,subsample=1,min_samples_split=2,min_samples_leaf=1,random_state=np.random.randint(2017))