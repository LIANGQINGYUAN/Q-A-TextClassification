# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:20:02 2018

@author: liang
"""

import numpy as np
import pandas as pd
import random
import operator
import collections

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
df_train=pd.read_csv('train.csv',encoding='gbk')
df_train=df_train.iloc[:,1:] #去掉第一列无关的数据
df_train['content']=pd.DataFrame(remove_uselessSymbol(df_train['content']))
df_train['question']=pd.DataFrame(remove_uselessSymbol(df_train['question']))

df_test=pd.read_csv('test_complete_w.csv',encoding='gbk')
df_test=df_test.iloc[:,1:] #去掉第一列无关的数据
df_test['content']=pd.DataFrame(remove_uselessSymbol(df_test['content']))
df_test['question']=pd.DataFrame(remove_uselessSymbol(df_test['question']))


'''
建立词典
'''
def bulid_mydict():
    code=int(0)
    mydict={}
    mydict['UNKNOW']=code
    code+=1
    
    #将训练集的中出现的词分开装进词表
    for i in df_train['content']:
        items=i.split(' ')
        for word in items:
            if not word in mydict:
                mydict[word]=code
                code+=1
    for i in df_train['question']:
        items=i.split(' ')
        for word in items:
            if not word in mydict:
                mydict[word]=code
                code+=1  
                
    #将测试集的中出现的词分开装进词表
    for i in df_test['content']:
        items=i.split(' ')
        for word in items:
            if not word in mydict:
                mydict[word]=code
                code+=1
    for i in df_test['question']:
        items=i.split(' ')
        for word in items:
            if not word in mydict:
                mydict[word]=code
                code+=1      
    return mydict
#mydict=bulid_mydict()
'''
建立词典的新方法
'''
#建立词典
vocabulary_size = 70000   # 词典大小
def build_dataset():
    #建立word集合
    words=[]
    for line in df_train['content']:
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
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary
#data, count, dictionary, reverse_dictionary = build_dataset()
#dictionary2 = build_dataset()


    
'''
获取答案对应的list
'''
def read_alist():
    return df_train['content'].tolist()
#alist=read_alist()
    
'''
获取标签为1的行
'''
def read_raw():
    raw=[]
    for i in range(len(df_train)):
        if int(df_train['label'][i]) is 1:
            raw.append(df_train.iloc[i,:])
    return raw
#raw=read_raw()

'''
读取用于评估的测试集
'''
def read_test():
    testlist=[]
    for i in range(len(df_test)):
        testlist.append(df_test.iloc[i,:])
    return testlist
#testlist=read_test()

'''
获取用于一次训练的数据

x_train_1 问题的编码   100X200 100-->抽取的数量    100-->设置每个问题/答案中出现的单词数量不超过100个
x_train_2 答案的编码
x_train_2 反例的编码
''' 
#定义一个随机抽取一行数据的函数-->抽取负样本
def rand_qa(raw,items):
    while(True):    
        index = random.randint(0, len(raw) - 1)
        #刚好抽取到当前问题所在的一行，则要保证label不为1
        if raw[index]['question'] == items['question'] :
            if raw[index]['label'] !=1:
                break
        #抽到其他问题的行，说明index选取符合条件
        if raw[index]['question'] != items['question'] :
            break
    return raw[index]['content']
#对数据进行编码的函数，size为假设的每个问题或答案不超过size个词
def encode_sent(mydict,string,size):
    x=[]
    words=string.split(' ')
    for i in range(size):
        if i<len(words) and words[i] in mydict:
            x.append(mydict[words[i]])
        else:
            x.append(mydict['UNKNOW'])
    return x
#加载数据方法,size为每次训练所需数据量
def load_data(mydict,alist,raw,size):
    x_train_1=[]
    x_train_2=[]
    x_train_3=[]
    for i in range(size):
        items=raw[random.randint(0,len(raw)-1)]
        nega=rand_qa(raw,items)#选取一个负例
        x_train_1.append(encode_sent(mydict,items['question'],300))
        x_train_2.append(encode_sent(mydict,items['content'],300))
        x_train_3.append(encode_sent(mydict,nega ,300))
    return np.array(x_train_1),np.array(x_train_2),np.array(x_train_3)
#x_train_1,x_train_2,x_train_3=load_data(mydict,alist,raw,100)
    
'''
获取用于评估的数据
'''
def load_data_val(testlist,mydict,index,batch):
    x_train_1=[]
    x_train_2=[]
    x_train_3=[]
    for i in range(batch):
        true_index=index+i
        if true_index>=len(testlist):
            true_index=len(testlist)-1
        x_train_1.append(encode_sent(mydict,testlist[true_index]['question'],300))
        x_train_2.append(encode_sent(mydict,testlist[true_index]['content'],300))
        x_train_3.append(encode_sent(mydict,testlist[true_index]['content'],300))
    return np.array(x_train_1),np.array(x_train_2),np.array(x_train_3)
#x_train_1,x_train_2,x_train_3=load_data_val(testlist,mydict,1,100)
    
'''
根据真实标签和文本得分判断正确率
'''
def calculate_match_rate(sessdict):
    lev1=0.0
    lev0=0.0
    for key,value in sessdict.items():
        
        #计算每个问题中label为1的个数
        sum_label=int(0)
        for _,i in sessdict[int(key)]:
            if i == 1:
               sum_label+=1
        
        #遍历
        value.sort(key=operator.itemgetter(0), reverse=True)
        
        if sum_label<=3:
            for i in range(sum_label):
                        score, label = value[i]
                        if label == 1:
                            lev1 += 1
                        if label == 0:
                            lev0 += 1
        else : 
            score, label = value[0]
            if label == 1:
                lev1 += 1
            if label == 0:
                lev0 += 1
    return lev1/(lev1+lev0)
#test_rate=calculate_match_rate(sessdict) 

  
        


    

    