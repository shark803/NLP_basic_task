# -*- coding: utf-8 -*-
# @Time : 2019/7/16 15:50
# @Author : Lei Wang
# @Site : 
# @File : textcnn_v2.py
# @descriptionn: use high level api in model_fn 
# @Software: vim
# ref url:https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import time
import datetime
import random
import json

from collections import Counter
from math import sqrt

import gensim
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import ConfigProto
from tensorflow import InteractiveSession
session_config = ConfigProto()
session_config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)
run_config = tf.estimator.RunConfig().replace(session_config=session_config)


tf.logging.set_verbosity(tf.logging.INFO)
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from config_textrnn_v1 import Config,TrainingConfig,ModelConfig
from data_helpers import Dataset



tf.logging.set_verbosity(tf.logging.INFO)

config = Config()
train_config = TrainingConfig()
model_config = ModelConfig()

data = Dataset(config)
data.dataGen()
data.TestdataGen()

print(tf.__version__)


wordEmbedding = data.wordEmbedding

def input_fn_maker(features,labels):
    def input_fn():
        '''        
        filenames = tfr.get_filenames(path=path, shuffle=shuffle)
        dataset=tfr.get_dataset(paths=filenames, data_info=data_info_path, shuffle = shuffle, 
                            batch_size = batch_size, epoch = epoch, padding =padding)
        '''
        dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat(train_config.epoches).batch(config.batchSize)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn


#test_input_fn = input_fn_maker('mnist_tfrecord/test/',  'mnist_tfrecord/data_info.csv',batch_size = 512, padding = padding_info)
#train_input_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', shuffle=True, batch_size = 128, padding = padding_info)
#train_eval_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', batch_size = 512, padding = padding_info)

train_input_fn = input_fn_maker(np.array(data.trainReviews, dtype="int64"),np.array(data.trainLabels, dtype="int32"))
train_eval_fn = input_fn_maker(np.array(data.evalReviews, dtype="int64"),np.array(data.evalLabels, dtype="int32"))
test_input_fn = input_fn_maker(np.array(data.testReviews, dtype="int64"),np.array([0]*len(data.testReviews), dtype="int32"))

def rnn_model_fn(features,labels,mode,params):

       
    def attention(config,H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = config.model.hiddenSizes[-1]
        
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))
        
        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)
        
        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))
        
        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, config.sequenceLength])
        
        # 用softmax做归一化处理[batch_size, time_step]
        alpha = tf.nn.softmax(restoreM)
        
        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, config.sequenceLength, 1]))
        
        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)
        
        sentenceRepren = tf.tanh(sequeezeR)
        
        # 对Attention的输出可以做dropout处理
        #output = tf.nn.dropout(sentenceRepren, dropoutKeepProb)
        
        return sentenceRepren
   

    #input layer

    #inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
    #inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")

    #dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

    #Convolutional Layer #1
    '''
    sentence = features['sentence']
    # Get word embeddings for each token in the sentence
    embeddings = tf.get_variable(name="embeddings", dtype=tf.float32, shape=[params["vocab_size"], FLAGS.embedding_size])
    sentence = tf.nn.embedding_lookup(embeddings, sentence) # shape:(batch, sentence_len, embedding_size)
    # add a channel dim, required by the conv2d and max_pooling2d method
    sentence = tf.expand_dims(sentence, -1) # shape:(batch
    '''
    with tf.name_scope("embedding"):
        # 利用预训练的词向量初始化词嵌入矩阵
        #embeddings  = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec") ,name="embedding")
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32, shape=[params['vocab_size'], params['embeddingSize']])
        # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
        embeddedWords = tf.nn.embedding_lookup(embeddings, features)
    
    dropoutKeepProb = 1.0
    if mode == tf.estimator.ModeKeys.TRAIN:
        dropoutKeepProb = 0.5
    # 定义两层双向LSTM的模型结构
    with tf.name_scope("Bi-LSTM"):
        for idx, hiddenSize in enumerate(config.model.hiddenSizes):
            with tf.name_scope("Bi-LSTM" + str(idx)):
                # 定义前向LSTM结构
                lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                                                                 output_keep_prob=dropoutKeepProb)
                # 定义反向LSTM结构
                lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                                                                 output_keep_prob=dropoutKeepProb)


                # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                outputs_, current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, 
                                                                                  embeddedWords, dtype=tf.float32,
                                                                                  scope="bi-lstm" + str(idx))
        
                # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                embeddedWords = tf.concat(outputs_, 2)
                
    # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
    outputs = tf.split(embeddedWords, 2, -1)
     
    # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
    with tf.name_scope("Attention"):
        H = outputs[0] + outputs[1]

        # 得到Attention的输出
        output = attention(config,H)
        #output = tf.contrib.rnn.AttentionCellWrapper(H, attn_length = params['attention_len']) # attention_len是自己指定的Attention关注长度
        outputSize = config.model.hiddenSizes[-1]

    with tf.name_scope("dropout"):
        hDrop = tf.nn.dropout(output, params['dropout_rate'])
    
    #logits = tf.layers.dense(hDrop, 2, activation=None)
    
    
  
    # 全连接层的输出
    
    with tf.name_scope("output"):
        outputW = tf.get_variable(
           "outputW",
            shape=[outputSize, 2],
            initializer=tf.contrib.layers.xavier_initializer())
        outputB= tf.Variable(tf.constant(0.1, shape=[2]), name="outputB")
        #l2Loss += tf.nn.l2_loss(outputW)
        #l2Loss += tf.nn.l2_loss(outputB)
        logits = tf.nn.xw_plus_b(hDrop, outputW, outputB, name="scores")
        #print(logits.shape)
        #logits = tf.expand_dims(logits, -1) 
        #logits = tf.nn.relu(scores, name="predictions")
        #self.predictions = tf.argmax(tf.nn.softmax(self.scores), 1, name="predictions")
        #self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.0), tf.int32, name="binaryPreds")


    
    
  

    
   
    
    predictions = {
        "classes": tf.argmax(input=logits, axis=1,name="classes"),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
    
    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)





    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #print(labels.shape, logits.shape)
    #exit(0)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def read_testfile(filename):
    with open(filename,'r',encoding='utf-8') as fin:
        x_raw = [line.rstrip().split('\t')[1] for line in fin.readlines()]
    return x_raw[1:]
x_raw = read_testfile("./data/rawData/testData.tsv")


imdb_classifier = tf.estimator.Estimator(model_fn=rnn_model_fn,
            params={
                'vocab_size':data.vocab_size,
                'attention_len':128,
                'embeddingSize': model_config.embeddingSize,
                'dropout_rate': 0.5
            },
            model_dir="imdb_model_textrnnv4",config=run_config)

imdb_classifier.train(input_fn=train_input_fn)

train_eval_results = imdb_classifier.evaluate(input_fn=train_eval_fn)
print('train set')
print(train_eval_results)


results = list(imdb_classifier.predict(input_fn=test_input_fn))
print('test set')
#print(results)
with open('./testResults_textrnn_v4.tsv','w',encoding='utf-8') as fout:
    for review,result in zip(x_raw,results):        
        fout.write(review+'\t'+str(result['classes'])+'\n')




