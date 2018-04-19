
from __future__ import print_function, division

import os
import os.path
import pandas as pd
from io import StringIO
import io
import unicodedata
import re

import tensorflow as tf
import numpy as np
np.set_printoptions(threshold = 10000)
import collections
import random

from gru import GRUCell as Cell #custom implementation with normalization
from tensorflow.python.ops.rnn import dynamic_rnn as rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from attention import attention
from bca import *
from ordloss import *
from utils import *
from dataUtils import *


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from sklearn.metrics import accuracy_score
 

# Read glove embeddings

filepath_glove = 'glove.6B.200d.txt'
folder_path = '/g/ssli/data/lingcomp'
#folder_path = 'embedding'
fp = os.path.join(folder_path, filepath_glove)

glove_vocab,  embedding_dict, embed_vector = load_emb_glove(fp)

glove_vocab_size = len(glove_vocab)
embedding_dim = len(embed_vector)

#read data; SEQUENCE_LENGTH is maximum length of sentence in words, SEQUENCE_LENGTH_D is maximum length of document in sentences. 

SEQUENCE_LENGTH = 65
SEQUENCE_LENGTH_D = 40
max_vocab = 20000
train_split = 0.85

#system parameters

HIDDEN_SIZE = 15
ATTENTION_SIZE = 10
HIDDEN_SIZE_D = 10
ATTENTION_SIZE_D = 5
KEEP_PROB = 0.4
BATCH_SIZE = 10
NUM_EPOCHS = 1  # max val_acc at __
DELTA = 0.75

#use ordinal regression; logistic regression if False
ordinal = False

directory = 'model'
if not os.path.exists(directory):
    os.makedirs(directory)

MODEL_PATH = "model/model%d" %(HIDDEN_SIZE + HIDDEN_SIZE_D)
#read train and val data
X_train, y_train, X_val, y_val, doc_vocab_size, embedding = read_test_train(glove_vocab, embedding_dict, embedding_dim, SEQUENCE_LEN = SEQUENCE_LENGTH, SEQUENCE_LEN_D = SEQUENCE_LENGTH_D, dname =  'bca%d'%(HIDDEN_SIZE + HIDDEN_SIZE_D), tr = train_split, max_vocab = max_vocab)

NUM_WORDS = doc_vocab_size
EMBEDDING_DIM = embedding_dim


print('Sentence length:',SEQUENCE_LENGTH)
print('Document length:',SEQUENCE_LENGTH_D)
print('Hidden size:',HIDDEN_SIZE)
print('Hidden size sentence level:',HIDDEN_SIZE_D)
print('Epochs:',NUM_EPOCHS)



# Sequences preprocessing
vocabulary_size = doc_vocab_size 

X_train = zero_pad(X_train, SEQUENCE_LENGTH)
X_val = zero_pad(X_val, SEQUENCE_LENGTH)


#batch size padding 
X_val = zero_pad_test(X_val, BATCH_SIZE*SEQUENCE_LENGTH_D)
y_val = zero_pad_test(y_val, BATCH_SIZE)


#Different placeholders
num_classes = y_train.shape[1]
batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH])
ind_list_ph = tf.placeholder(tf.int32, [None])
target_ph = tf.placeholder(tf.float32, [None,num_classes])
seq_len_ph = tf.placeholder(tf.int32, [None])
seq_len_ph_d = tf.placeholder(tf.int32, [None])
keep_prob_ph = tf.placeholder(tf.float32)


# Embedding layer
embeddings_var = tf.Variable(tf.random_uniform([vocabulary_size, EMBEDDING_DIM], -1.0, 1.0), trainable=True)
batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

# (Bi-)RNN layer(-s)
with tf.variable_scope('sentence'):
    rnn_outputs, _ = bi_rnn(Cell(HIDDEN_SIZE), Cell(HIDDEN_SIZE), inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
    rnn_outputs = cross_attention(rnn_outputs, SEQUENCE_LENGTH_D, seq_len_ph, BATCH_SIZE, time_major=False, return_alphas=False)
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, seq_len_ph, return_alphas=True)
    attention_output = tf.reshape(attention_output,[BATCH_SIZE, SEQUENCE_LENGTH_D, HIDDEN_SIZE*2*3])
    
with tf.variable_scope('document'):
    rnn_outputs_d, _ = bi_rnn(Cell(HIDDEN_SIZE_D), Cell(HIDDEN_SIZE_D), inputs=attention_output, sequence_length=seq_len_ph_d, dtype=tf.float32)
    attention_output_d, alphas_d = attention(rnn_outputs_d, ATTENTION_SIZE_D, seq_len_ph_d, return_alphas=True)

# Dropout
drop = tf.nn.dropout(attention_output_d, keep_prob_ph)

if ordinal:
    # For ordinal regression, same weights for each class
    W = tf.Variable(tf.truncated_normal([drop.get_shape()[1].value], stddev=0.1))
    W_ = tf.transpose(tf.reshape(tf.tile(W,[num_classes - 1]),[num_classes - 1, drop.get_shape()[1].value]))
    b = tf.Variable(tf.cast(tf.range(num_classes - 1), dtype = tf.float32))
    y_hat_ = tf.nn.xw_plus_b(drop, tf.negative(W_), b)

    # Predicted labels and logits
    y_preds, logits = preds(y_hat_,BATCH_SIZE)
    y_true = tf.argmax(target_ph, axis = 1)

    # Ordinal loss
    loss = ordloss_m(y_hat_, target_ph, BATCH_SIZE)
    c = stats.spearmanr
    str_score = "Spearman rank:"

else:
    W = tf.Variable(tf.truncated_normal([drop.get_shape()[1].value, num_classes], stddev=0.1))  
    b = tf.Variable(tf.constant(0., shape=[num_classes]))
    y_hat_ = tf.nn.xw_plus_b(drop, W, b)
    #y_hat_ = tf.squeeze(y_hat)
    # Cross-entropy loss and optimizer initialization
    y_preds = tf.argmax(y_hat_, axis = 1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat_, labels=target_ph))
    c = accuracy_score
    str_score = "Accucary:"

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
# Batch generators
train_batch_generator = batch_generator(X_train, y_train, BATCH_SIZE, seq_len = SEQUENCE_LENGTH_D)
val_batch_generator = batch_generator(X_val, y_val, BATCH_SIZE, seq_len = SEQUENCE_LENGTH_D)


saver = tf.train.Saver()

if __name__ == "__main__":
    config = tf.ConfigProto(inter_op_parallelism_threads=6,
                            intra_op_parallelism_threads=6)
    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(embeddings_var.assign(embedding))

        print("Start learning...")
        for epoch in range(NUM_EPOCHS):
            loss_train = 0
            loss_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training
            num_batches = X_train.shape[0] // (BATCH_SIZE*SEQUENCE_LENGTH_D)
            for bx in range(num_batches*2):
                x_batch, y_batch = next(train_batch_generator)
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                seq_len_d = []               
                l = SEQUENCE_LENGTH_D
                for i in range(0,len(x_batch),l):
                    for j in range(i,i+l):
                        if list(x_batch[j]).index(0) == 0:
                            seq_len_d.append(j%l)
                            break
                        elif j == i+l-1:
                            seq_len_d.append(l)

                seq_len_d = np.array(seq_len_d)

                loss_tr,  _ = sess.run([loss,  optimizer],
                                           feed_dict={batch_ph: x_batch,
                                                      target_ph: y_batch,
                                                      seq_len_ph: seq_len,
                                                      seq_len_ph_d: seq_len_d,
                                                      keep_prob_ph: KEEP_PROB})
                loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
            
             
            print('training loss: ' + str(loss_train))

            #testing on the validation set
            num_batches = X_val.shape[0] // (BATCH_SIZE*SEQUENCE_LENGTH_D)
            true = []
            preds = []

            for bx in range(num_batches):
                x_batch, y_batch = next(val_batch_generator)
                seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
                seq_len_d = []               
                l = SEQUENCE_LENGTH_D
                for i in range(0,len(x_batch),l):
                    for j in range(i,i+l):
                        if list(x_batch[j]).index(0) == 0:
                            seq_len_d.append(j%l)
                            break
                        elif j == i+l-1:
                            seq_len_d.append(l)

                seq_len_d = np.array(seq_len_d)

                pred,loss_t = sess.run([y_preds,loss],
                              feed_dict={batch_ph: x_batch,
                                    target_ph: y_batch,
                                    seq_len_ph: seq_len,
                                    seq_len_ph_d: seq_len_d,
                                    keep_prob_ph: 1.0})
                preds.extend(pred)
                t = np.argmax(y_batch, axis = 1)
                true.extend(t)

            spr = c(true, preds)
            if ordinal:
                spr = spr[0]
            print('Validation set '+ str_score + str(spr))


        saver.save(sess, MODEL_PATH)
        print("saved at" + MODEL_PATH)

