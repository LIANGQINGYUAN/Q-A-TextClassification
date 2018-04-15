# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 14:55:11 2018

@author: liang
"""

import pandas as pd

#dispose train data
df=pd.read_json('train_data_sample.json',encoding='utf-8')
df_question=pd.DataFrame()
df_question['item_id']=df['item_id']
df_question['question']=df['question']
df_question.to_csv('df_train_question.csv')

#split train passages to row 
mysplit=pd.DataFrame()
for i in range(len(df)):
    df_passages=pd.DataFrame(df['passages'][i]).to_json()#get passages of each question
    df_splited=pd.read_json(df_passages)                 #split json of passages
    df_splited['item_id']=df['item_id'][i]               #add 'item_id' column
    df_splited['question']=df['question'][i]
    mysplit=mysplit.append(df_splited)                   #append data if each loop

mysplit.to_csv('train.csv')

#dispose test data
df_test=pd.read_json('test_data_sample.json',encoding='utf-8')
df_test_question=pd.DataFrame()
df_test_question['item_id']=df_test['item_id']
df_test_question['question']=df_test['question']

#df_test_question.to_csv('df_test_question.csv')
#split train passages to row 
mysplit_test=pd.DataFrame()
for i in range(len(df_test)):
    df_test_passages=pd.DataFrame(df_test['passages'][i]).to_json()  #get passages of each question
    test_splited=pd.read_json(df_test_passages)                      #split json of passages
    test_splited['item_id']=df_test['item_id'][i]               #add 'item_id' column
    test_splited['question']=df_test['question'][i]
    mysplit_test=mysplit_test.append(test_splited)                #append data if each loop
    
mysplit_test.to_csv('test_complete.csv')