# -*- coding: utf-8 -*-
# @Time : 2019/7/15 15:12
# @Author : Lei Wang
# @Site : 
# @File : textcnn_v4.py
# @descriptionn: use tf.nn in model_fn 
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
config = ConfigProto()
config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)
run_config = tf.estimator.RunConfig().replace(session_config=config)


tf.logging.set_verbosity(tf.logging.INFO)
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from config_textcnn_v1 import Config,TrainingConfig
from data_helpers import Dataset



tf.logging.set_verbosity(tf.logging.INFO)

config = Config()
train_config = TrainingConfig()

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

def cnn_model_fn(features,labels,mode):

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
        embeddings  = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec") ,name="embedding")
        #embeddings = tf.get_variable(name="embeddings", dtype=tf.float32, shape=[data.vocab_size, 200])
        # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
        embeddedWords = tf.nn.embedding_lookup(embeddings, features)
        # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
        embeddedWordsExpanded = tf.expand_dims(embeddedWords,-1)
    # 创建卷积和池化层
    training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_emb = tf.layers.dropout(inputs=embeddedWords, 
                                    rate=0.2, 
                                    training=training)

    conv = tf.layers.conv1d(
        inputs=dropout_emb,
        filters=32,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)
    
    # Global Max Pooling
    pool = tf.reduce_max(input_tensor=conv, axis=1)
    
    hidden = tf.layers.dense(inputs=pool, units=250, activation=tf.nn.relu)
    
    dropout_hidden = tf.layers.dropout(inputs=hidden, 
                                       rate=0.2, 
                                       training=training)
    
    logits = tf.layers.dense(inputs=dropout_hidden, units=2)
    '''
    pooledOutputs = []
    # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
    for i, filterSize in enumerate(config.model.filterSizes):
        with tf.name_scope("conv-maxpool-%s" % filterSize):
            # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
            # 初始化权重矩阵和偏置
            filterShape = [filterSize, config.model.embeddingSize, 1, config.model.numFilters]
            W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]), name="b")
            conv = tf.nn.conv2d(
                embeddedWordsExpanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")

            # relu函数的非线性映射
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # 池化层，最大池化，池化是对卷积后的序列取一个最大值
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, config.sequenceLength - filterSize + 1, 1, 1],  # ksize shape: [batch, height, width, channels]
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中
            

    # 得到CNN网络的输出长度
    numFiltersTotal = config.model.numFilters * len(config.model.filterSizes)

    # 池化后的维度不变，按照最后的维度channel来concat
    hPool = tf.concat(pooledOutputs, 3)
    # 摊平成二维的数据输入到全连接层
    hPoolFlat = tf.reshape(hPool, [-1, numFiltersTotal])

    keep_prob = 1.0
    if mode == tf.estimator.ModeKeys.TRAIN:
        keep_prob = 0.5

    with tf.name_scope("dropout"):
        hDrop = tf.nn.dropout(hPoolFlat, keep_prob)
    
    #logits = tf.layers.dense(hDrop, 2, activation=None)
    
    
  
    # 全连接层的输出
    
    with tf.name_scope("output"):
        outputW = tf.get_variable(
           "outputW",
            shape=[numFiltersTotal, 2],
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
    
   '''
    
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


imdb_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="imdb_model_textcnnv4",config=run_config)

imdb_classifier.train(input_fn=train_input_fn)

train_eval_results = imdb_classifier.evaluate(input_fn=train_eval_fn)
print('train set')
print(train_eval_results)


results = list(imdb_classifier.predict(input_fn=test_input_fn))
print('test set')
#print(results)
with open('./testResults_modlev4.tsv','w',encoding='utf-8') as fout:
    for review,result in zip(x_raw,results):        
        fout.write(review+'\t'+str(result['classes'])+'\n')




