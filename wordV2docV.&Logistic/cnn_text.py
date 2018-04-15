# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 23:14:52 2018

@author: liang
"""

import tensorflow as tf
import pandas as pd
import numpy as np
'''
读取数据
'''
def read_data():
    df=pd.read_csv('train_vec_test.csv')
    df_vec=df.iloc[:,1:]
    return df_vec

df_vec=read_data()

'''
打乱数据
'''
from sklearn.utils import shuffle 
df_vec=shuffle(df_vec)

#sess = tf.InteractiveSession()
sess = tf.Session()

'''
定义CNN结构
'''
#权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#卷积核和池化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')  
def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')  
    
#输入变量                        
x = tf.placeholder(tf.float32, [None, 324])
y_ = tf.placeholder(tf.float32, [None,1])
x_image = tf.reshape(x, [-1,18,18,1])
#第一层卷积                        
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_3x3(h_conv2)
#连接层
#加入一个有512个神经元的全连接层，用于处理整个图片。
W_fc1 = weight_variable([3*3*64, 20])
b_fc1 = bias_variable([20])
h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#为了减少过拟合，我们在输出层之前加入dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#输出层
W_fc2 = weight_variable([20, 1])
b_fc2 = bias_variable([1])
y_conv=tf.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#初始化变量
sess.run(tf.global_variables_initializer())

'''
分割训练数据
'''
def split_data(df_vec):
    train=df_vec.iloc[:int(len(df_vec)*0.7),:]
    test=df_vec.iloc[int(len(df_vec)*0.7):,:]
    return train,test
'''
获取每次训练使用的数据
'''
def get_batch(df_vec,batch_start,batch_each):
    max_index=len(df_vec)
    batch_end=batch_start+batch_each
    
    x_batch=df_vec.iloc[batch_start:batch_end,:324]
    y_batch=np.array(df_vec['label'][batch_start:batch_end]).reshape(-1,1)
    
    batch_start+=batch_each
    batch_end+=batch_each
    if batch_end>=max_index:
        batch_end=max_index-1
    
    return x_batch,y_batch

'''
训练
'''
def train(df_vec):
    mypred=[]
    start=0
    each=100
    for i in range(100):
        x_batch,y_batch=get_batch(df_vec,start,each)
        if i%10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    x:x_batch, 
                    y_:y_batch,
                    keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        start+=each
        #sess.run需要有fetchs参数
        mypred_temp=sess.run(y_conv,feed_dict={ x:x_batch, y_:y_batch , keep_prob: 0.5})
        mypred.append(mypred_temp)
    
    return mypred

'''
测试
'''
def test(df_vec):
    mypred_test=[]
    start=0
    each=100
    for i in range(3000):
        x_batch,y_batch=get_batch(df_vec,start,each)
        if i%10 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    x:x_batch, 
                    y_:y_batch,
                    keep_prob: 1.0})
            print("step %d, test accuracy %g"%(i, train_accuracy))
        start+=each
        #sess.run需要有fetchs参数
        mypred_temp=sess.run(y_conv,feed_dict={ x:x_batch, y_:y_batch , keep_prob: 0.5})
        mypred_test.append(mypred_temp)

    return mypred_test

'''
主程序
'''
with tf.Graph().as_default():
    with tf.device("/cpu:0"):
        with sess.as_default(): 
            df_train,df_test=split_data(df_vec)
            mypred=train(df_train)     
            #mypred_test=test(df_test)