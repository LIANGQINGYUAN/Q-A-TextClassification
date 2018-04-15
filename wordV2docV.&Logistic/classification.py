# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:58:45 2018

@author: liang
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split , cross_val_score 
'''
读取数据
'''
def read_data():
    df=pd.read_csv('train_vec_2.csv')
    df_vec=df.iloc[:,1:]
    return df_vec

'''
分割训练数据
'''
def split_data(df_vec):
    x=df_vec.iloc[:,:324]
    y=df_vec['label']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42) 
    return X_train, X_test, y_train, y_test
'''
计算分类精度
'''  
from sklearn import metrics  
def metrics_result(actual, predict):  
    print ('精度:{0:.3f}'.format(metrics.precision_score(actual, predict,average='weighted')) ) 
    print ('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict,average='weighted'))   )
    print ('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict,average='weighted'))   )

'''
逻辑回归
'''
def logistic(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.linear_model import RandomizedLogisticRegression as RLR
    #特征工程
    rlr=RLR()
    rlr.fit(X_train,y_train)
    print(rlr.get_support())
    x=X_train[X_train.columns[rlr.get_support()]].as_matrix()
    x_test=X_test[X_test.columns[rlr.get_support()]].as_matrix()
    '''
    x=X_train
    x_test=X_test
    '''
    #逻辑回归
    lr=LR()
    lr.fit(x,y_train)   
    pred_prob_train = lr.predict_proba(x) 
    pred_prob = lr.predict_proba(x_test)  
    print('logistic')  
    predicts=lr.predict(x_test)
    metrics_result(y_test,predicts)

    return pred_prob,pred_prob_train

'''
SVM
'''
def svm(X_train, X_test, y_train, y_test):
    
    from sklearn.svm import SVC
    svc = SVC(probability=True)
    svc.fit(X_train,y_train) 
    pred_prob_svc = svc.predict_proba(X_test)  
    pred_prob_svc_train = svc.predict_proba(X_train) 
    scores = cross_val_score(svc,X_train,y_train,cv=5)  
    print('svm')
    print ('训练准确率：',np.mean(scores),scores)  
    print('测试准确率：',svc.score(X_test,y_test))
    
    return pred_prob_svc,pred_prob_svc_train

'''
手动设置阈值进行测试
'''
def  mythreshold_predict(pred_prob, threshold,y):
    mypredict=[]
    for i in pred_prob:
        if i[1]>threshold:
            mypredict.append(1)
        else:
            mypredict.append(0)
    print('阈值：%.4f'%threshold)       
    metrics_result(y,mypredict)
    
    '''
    比较相似度
     
    lev1=0
    lev0=0  
    for i in  range(len(mypredict)):
        if mypredict[i] == list(y)[i]:
            lev1+=1
        else:
            lev0+=1
    print('阈值：%.4f  ;修改之后的准确率：%.4f'%(threshold,lev1/(lev1+lev0)))
    '''  

df_vec=read_data()
X_train, X_test, y_train, y_test = split_data(df_vec)
pred_prob,pred_prob_train=logistic(X_train, X_test, y_train, y_test)
#pred_prob_svc,pred_prob_svc_train=svm(X_train, X_test, y_train, y_test)
threshold=0.42
print('----------训练----------')
mythreshold_predict(pred_prob_train,threshold,y_train)
print('----------测试----------')
mythreshold_predict(pred_prob,threshold,y_test)
'''
while(threshold<0.5):
    print('-----训练-----')
    mythreshold_predict(pred_prob_train,threshold,y_train)
    print('-----测试-----')
    mythreshold_predict(pred_prob,threshold,y_test)
    threshold+=0.01
'''
