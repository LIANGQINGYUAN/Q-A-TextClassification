# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:51:54 2018

@author: liang
"""
import pandas as pd

'''
对数据进行分词
'''

#df_train = pd.read_csv('train.csv',encoding='gbk')
#df_train=df_train.iloc[:,1:] #去掉第一列无关的数据
df_test=pd.read_csv('test.csv',encoding='gbk')
df_test=df_test.iloc[:,1:] #去掉第一列无关的数据

def cut_stopWords(df,feature):
    
    #进行分词
    import jieba
    mycut=lambda s:' '.join(jieba.cut(s))
    
    df[feature] = df[feature].astype('str')
    documents=df[feature].apply(mycut)
    '''
    去停用词
    '''
    import codecs
    with codecs.open("stopwords3.txt", "r", encoding="gbk") as f:
        text = f.read()
    stoplists=text.splitlines()

    texts = [[word for word in document.split()if word not in stoplists] for document in documents]

    
    
    #去掉一些没用的符号
    list_str_texts=[]
    for i in range(len(texts)):
        s=str(texts[i])
        s=s.replace('[','').replace(']','')
        list_str_texts.append(s)
    df[feature]=pd.DataFrame(list_str_texts)
    
    #df[feature]=pd.DataFrame(texts)
    return texts

#cut_stopWords(df_train,'content')
#cut_stopWords(df_train,'question')
#df_train.to_csv('train_w.csv')
cut_stopWords(df_test,'content')
cut_stopWords(df_test,'question')
df_test.to_csv('test_complete_w.csv')
