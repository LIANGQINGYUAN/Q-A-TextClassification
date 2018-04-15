# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 23:20:39 2018

@author: liang
"""
import tensorflow as tf
import numpy as np
import os 
import sys 
import time
import datetime
import data_helper
from CNN import InsQACNN
import pandas as pd
import operator

'''
设置参数
'''
#超参数设置
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.25, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")#50000
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")#3000
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")#3000
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS(sys.argv) 
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

'''
读取数据
'''
print("Loading data...")

#mydict = data_helper.bulid_mydict() #问题所对应的单词表
mydict = data_helper.build_dataset() 
alist = data_helper.read_alist()  #答案对应的列 
raw = data_helper.read_raw()      #标签为1的行
x_train_1, x_train_2, x_train_3 = data_helper.load_data(mydict, alist, raw, FLAGS.batch_size)#选取数量为size的集合，编码后返回
testList=data_helper.read_test()
print('x_train_1', np.shape(x_train_1))
print("Load done...")

'''
训练
'''
with tf.Graph().as_default():
    with tf.device("/cpu:0"):
      session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
      sess = tf.Session(config=session_conf)
      with sess.as_default():
        cnn = InsQACNN(
            sequence_length=x_train_1.shape[1],
            batch_size=FLAGS.batch_size,
            vocab_size=len(mydict),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        '''
        定义训练程序
        '''
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-1)
        #optimizer = tf.train.GradientDescentOptimizer(1e-2)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step) 
        '''
        跟踪梯度和稀疏性
        '''
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        
        '''
        输出模型以及摘要
        '''
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        '''
        初始化变量
        '''
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        '''
        训练
        
        打印并保存训练的结果
        '''
        def train_step(x_batch_1, x_batch_2, x_batch_3):
            """
            A single training step
            """
            #feed_dict参数，将数据传入sess.run()函数。
            feed_dict = {
              cnn.input_x_1: x_batch_1,
              cnn.input_x_2: x_batch_2,
              cnn.input_x_3: x_batch_3,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
        
        '''
        评估
        '''
        scoreList = []  #得分
        sessdict = {}   #用于计算预测准确率  结构：{qid：[(0.886,1),(0.647,0),(0.647,0).....]}
        def dev_step():
          #循环计算每一句与问题的相似程度
          i = int(0)
          while True: 
              #x_test_2, x_test_3是一样的
              x_test_1, x_test_2, x_test_3 = data_helper.load_data_val(testList, mydict, i, FLAGS.batch_size)
              feed_dict = {
                cnn.input_x_1: x_test_1,
                cnn.input_x_2: x_test_2,
                cnn.input_x_3: x_test_3,
                cnn.dropout_keep_prob: 1.0
              }
              batch_scores = sess.run([cnn.cos_12], feed_dict)
              for score in batch_scores[0]:
                  scoreList.append(score)
              i += FLAGS.batch_size
              if i >= len(testList):
                  break
          #passage_id   
          index=int(0)
          for item in testList:
              qid=int(item['item_id'])
              if not qid in sessdict:
                  sessdict[qid]=[]
              sessdict[qid].append((scoreList[index],int(item['passage_id'])))
              index+=1
              if index>=len(testList):
                  break
              
          #将score和passage写入文件
          pred_score=[]
          p_id_list=[]
          temp_score_list=[]
          label_list=[]
          for key,value in sessdict.items():
              for score,passage_id in sessdict[int(key)]:
                  pred_score.append(score)
                  temp_score_list.append(score)
                  p_id_list.append(passage_id)
                  
                  if score>0.7:
                      label_list.append(1)
                  else:
                      label_list.append(0)
              #给相似度最高的标记为1
              #sessdict[int(key)].sort(key=operator.itemgetter(0), reverse=True) 
              #for i in range(3):
          
          #最终提交的label1数据
          df_score=pd.DataFrame()
          df_score['pred_score']=pred_score
          df_score['p_id_list']=p_id_list
          df_score['label_list']=label_list
          df_score.to_csv('df_score.csv')
          
          #打印准确率
          #print(data_helper.calculate_match_rate(sessdict))
        '''
        对每个数据块进行训练的循环
        '''
        # Generate batches
        # Training loop. For each batch...
        for i in range(FLAGS.num_epochs):
            try:
                x_batch_1, x_batch_2, x_batch_3 = data_helper.load_data(mydict, alist, raw, FLAGS.batch_size)
                train_step(x_batch_1, x_batch_2, x_batch_3)
                current_step = tf.train.global_step(sess, global_step) 
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step()
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            except Exception as e:
                print(e)
