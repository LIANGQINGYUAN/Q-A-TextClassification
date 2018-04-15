# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 21:31:55 2018

@author: liang
"""
import collections
import numpy as np
import pandas as pd

#word2vec
from gensim.models import Word2Vec

#去掉一些没用的符号的函数
def remove_uselessSymbol(texts):
    #去掉一些没用的符号
    list_str_texts=[]
    for i in range(len(texts)):
        s=str(texts[i])
        s=s.replace('[','').replace(']','').replace("'",'').replace(",",'').replace("+",'').replace("#",'').replace("&",'').replace("..",'')
        list_str_texts.append(s)
    return list_str_texts

'''
读取数据
'''
def read_data(dataframe,feature):
    orign_data=dataframe[feature]
    words=[]
    for line in orign_data:
        if type(line) != float:
            words.append(line.replace('[','').replace(']','').replace("'",'').replace(",",'').replace("+",'').replace("#",'').replace("&",'').replace("..",'').strip().split())
        else:
             words.append(str(line))
    return words

'''
建立词典
'''
def build_dataset(df,feature):
    vocabulary_size = 100000   # 词典大小
    #建立word集合
    words=[]
    for line in df[feature]:
        if type(line) != float:
            for word in line.strip().split():
                words.append(word)
    #建立词典          
    count = [['UNKNOW', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1)) #50000大小的词汇表
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    # 超过100000的词变成unk
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    #反字典
    #reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary

'''
转化为词向量
'''
def word2vec(texts):
    model = Word2Vec(texts, 
                 size=162, 
                 window=5,
                 min_count=2, 
                 workers=2)
    
    vocab = model.wv.vocab
    #print(model.wv['高速'])
    return model,vocab
    '''
    df_wordV =pd.DataFrame()
    words=[]
    i=1
    reverse_dictionary={}
    for word in vocab:
        print(i)
        words.append(word)
        #insert第一个参数代表插入列位置
        reverse_dictionary[word]=model[word]
        df_wordV.insert( i-1,i,model[word])
        i=i+1
    return df_wordV
    '''
    
'''
转化为句向量
'''
def wordv2docv(model,vocab,words,df_size):
    doc_vec_list=[]
    doc_vec__q_list=[]
    i=0
    for item in words:#一句话
        doc_vec=np.zeros(162)
        length=len(item) #一句话的单词数量
        #每句话的单词向量叠加
        for index in range(length):
            if item[index] in vocab:
                doc_vec+=model.wv[item[index]]
            else :
                doc_vec+=(np.zeros(162)+0.001)
        if length!=0:
            doc_vec=doc_vec/length
        #前一半给回答，后一半给问题
        if i<int(df_size/2):
            doc_vec_list.append(np.array(doc_vec))
        else:
            doc_vec__q_list.append(np.array(doc_vec))
        
        i+=1
    return doc_vec_list,doc_vec__q_list

'''
读取数据
'''
df_train=pd.read_csv('train.csv',encoding='gbk')
words=read_data(df_train,'content')
words_question=read_data(df_train,'question')

words.extend(words_question)
#df_wordV=word2vec(words)
#df_wordV.to_csv('word_v')


'''
词向量
'''
model,vocab=word2vec(words)
model.save('word2vec.model')

#model_question,vocab_question=word2vec(words)
#model_question.save('word2vec_q.model')
'''
生成句向量
'''
doc_vec_list,doc_vec__q_list=wordv2docv(model,vocab,words,len(words))
#doc_vec_list_question=wordv2docv(model_question,vocab_question,words_question,'question')


'''
存储数据
'''
#训练集
df_docv=pd.DataFrame()
df_docv_q=pd.DataFrame()
#df_docv['c_vec']=
#转化为每个数字之后到DataFrame
df_docv=df_docv.append(doc_vec_list)
df_docv_q=df_docv_q.append(doc_vec__q_list)   
#连接问题和答案
result=pd.concat([df_docv,df_docv_q],axis=1)

result['passage_id']=df_train['passage_id']
result['label']=df_train['label']


result.to_csv('train_vec_2.csv')