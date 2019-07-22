# -*- coding: utf-8 -*-
# @Time : 2019/7/11 10:30
# @Author : Lei Wang
# @Site : 
# @File : eval_textcnn_v1.py
# @Software: vim

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn
import csv
import math
from tensorflow.metrics import accuracy,auc,precision,recall
from config_textrnn_v1 import Config
from data_helpers import Dataset

# Parameters
# ==================================================

# Data Parameters

config = Config()

data = Dataset(config)
data.TestdataGen()

# 训练模型

# 生成训练集和验证集
testReviews = data.testReviews


#wordEmbedding = data.wordEmbedding


def TestBatchGen(x,batchsize):
    numBatches = len(x) // batchsize
    for i in range(numBatches):
        start = i*batchsize
        end = start+batchsize
        batchX = np.array(x[start: end], dtype="int64")
        yield batchX


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./model/textRNN_v1/Checkpoints", "Checkpoint directory")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("test_file", "./data/rawData/testData.tsv", "path of the IMDB test file")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")


def read_testfile(filename):
    with open(filename,'r',encoding='utf-8') as fin:
        x_raw = [line.rstrip().split('\t')[1] for line in fin.readlines()]
    return x_raw[1:]
x_raw = read_testfile(FLAGS.test_file)

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    #x_raw  = read_testfile(FLAGs.test_file)
    x_test= testReviews
    #y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ['I love this movie']
    y_test = [1]

print("\nx_test:")
print(x_test)
print("\nEvaluating...\n")
label_map={1:'cls:love',0:'cls:hate'}

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(inter_op_parallelism_threads=os.cpu_count(),
                        intra_op_parallelism_threads=os.cpu_count(),
                        log_device_placement=True)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("inputX").outputs[0]
        # input_y = graph.get_operation_by_name("inputY").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/binaryPreds").outputs[0]

        # Generate batches for one epoch
        batches = TestBatchGen(x_test,FLAGS.batch_size)
        

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            #print(batch_predictions)
            all_predictions.extend(batch_predictions)

with open('testResults_rnn_v1.tsv','w',encoding='utf-8') as fout:
    for review,label in zip(x_raw,all_predictions):        
        fout.write(review+'\t'+label_map[label[0]]+'\n')

