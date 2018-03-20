#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attribute to Sequence Language Model
"""
import argparse
import sys
import time
from datetime import datetime
from util import Progbar
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import copy
from model import Model
from util import minibatches
import os,pickle
import sklearn.metrics
from lstm import RNNModel

class Attribute2SequenceModel(RNNModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will generate wine reviews given attributes (region, points, price, etc.)
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.
        """
        self.input_placeholder = tf.placeholder(tf.int32, shape = (None, self.max_length, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape = (None, self.max_length))
        self.mask_placeholder = tf.placeholder(tf.bool, shape = (None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape = ())
        self.attribute_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_attributes))

    def add_attribute_embedding(self):
        embeddings = tf.get_variable("attribute_embeddings", shape=(self.n_attribute_classes, self.attribute_embed_size),initializer = tf.contrib.layers.xavier_initializer())
        embeddings = tf.embedding_looking(embeddings, self.attribute_placeholder)


    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        #with tf.variable_scope("RNN", reuse = tf.AUTO_REUSE):
        embeddings = tf.get_variable("embeddings", initializer = self.pretrained_embeddings,trainable=True)
        embeddings = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        embeddings = tf.reshape(embeddings, [-1, self.max_length, self.config.n_features* self.config.embed_size])
        embeddings = tf.cast(embeddings, tf.float32)
        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled RNN.
        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        U = tf.get_variable("OutputWeights", shape = (self.config.hidden_size, self.config.n_classes), initializer = tf.contrib.layers.xavier_initializer())
        b_2 = tf.get_variable("OutputBias", shape = (self.config.n_classes), initializer = tf.zeros_initializer())
        
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size, initializer = tf.contrib.layers.xavier_initializer()) for size in [self.config.hidden_size, self.config.hidden_size]] 
        #runs the entire rnn - "state" is the final state of the lstm
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers) 
        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=x, dtype=tf.float32)
        outputs = tf.nn.dropout(outputs, dropout_rate) 

        if self.many2one:
            mask = tf.cast(self.mask_placeholder, dtype = tf.float32)
            outputs_copy = tf.multiply(outputs, tf.expand_dims(mask, 2)) 
            outputs = tf.reduce_mean(outputs_copy, axis = 1)
        else:
            outputs = tf.reshape(outputs, [-1, self.config.hidden_size])    
        preds = tf.add(tf.matmul(outputs, U), b_2)
        if not self.many2one:
            preds = tf.reshape(preds, [-1, self.config.max_length, self.config.n_classes]) 
        #preds = tf.Print(preds, [preds], summarize = self.config.n_classes)
        return preds
    
    def evaluate(self, sess, examples_raw, examples = None):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
        sess: the current TensorFlow session.
        examples: A list of vectorized input/output pairs.
        examples_raw: A list of the original input/output sequence pairs.
        Returns:
        The 1-to-1 accuracy for predicting attributes, F_1 (weighted), F_1 (macro)
        """
        #token_cm = ConfusionMatrix(labels=LBLS)
        def accuracy_score(Y_pred, Y_true):
            '''
            returns: array of accuracy scores of size n_attributes or batch_sze depending on axis
            '''
            acc_array = np.array([])
            for pred, true in zip(Y_pred, Y_true):
                accuracy = np.array(pred)==np.array(true)
                acc_array = np.append(acc_array,np.mean(accuracy))
            return np.mean(acc_array)
        
        def f1_score(Y_pred, Y_true, average = 'weighted'):
            f1_scores = np.array([])
            for col_pred, col_true in zip(Y_pred.T, Y_true.T):
                f1_scores =  np.append(f1_scores, sklearn.metrics.f1_score(col_pred, col_true, average=average))    
            return f1_scores

        def test_accuracy(Y_pred,Y_true):
            acc_batch = np.mean(accuracy_score(Y_pred, Y_true))
            f1_w = 0
            f1_m = 0
            #f1_w  = np.mean(f1_score(Y_pred,Y_true,average="weighted"))  
            #f1_m = np.mean(f1_score(Y_pred,Y_true,average="macro"))  
            return acc_batch,f1_w,f1_m

        acc_array = []
        sentences, class_labels, predictions = zip(*self.output(sess, examples_raw, examples))
        return test_accuracy(predictions,class_labels)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            labels_ = None
            #print("labels:", labels)
            #print("preds unmasked:", preds[i])
            labels_gt = labels[:]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            #print("preds:", labels_)
            assert len(labels_) == len(labels_gt)
            #print("labels np:", np.array(labels))
            #print(" ")
            ret.append([sentence, labels_gt, labels_])
        #print("Predictions (sent, true, pred): ", ret)
        return ret

    def save_model_description(self):
        '''
        Saves the following information:  
        - Data Limit
        - Test Batch
        - Config
        '''
        with open(self.config.desc_output,"wb+") as f:
            with open(self.config.desc_txt,"w+") as g:
                all_info = {}
                all_info["config"] = {a:getattr(self.config,a) for a in dir(self.config) if not a.startswith('__') and not a=="update_outputs"}
                all_info["limit"] = self.limit
                all_info["test_batch"] = self.test_batch
                pickle.dump(all_info,f)

    def save_epoch_outputs(self,epoch,loss,result_dev,result_train):
        '''
        Saves each epoch's output to the csv. Note that opens and closes CSV every time, so can track what is happening
        even with screen 
        '''
        #if Y_cat == None: Y_cat = self.cat
        if not os.path.exists(self.config.epochs_csv):
            with open(self.config.epochs_csv,"w+") as f:
                f.write("Limit,"+str(self.limit)+"\n")
                f.write("LR,"+str(self.config.lr)+"\n")
                f.write("HS,"+str(self.config.hidden_size)+"\n")
                f.write("Loss,dev_ACC,dev_F1_W,dev_F1_M,train_ACC,train_F1_W,train_F1_M,epoch\n")

        with open(self.config.epochs_csv,"a") as f:
            f.write(str(loss)+",")
            for r in result_dev:f.write(str(r)+",") 
            for r in result_train:f.write(str(r)+",") 
            f.write(str(epoch)+"\n")



    def __init__(self, helper, config, pretrained_embeddings,cat=None,test_batch=None,limit=None, many2one=False):
        print("Num Classes: ",config.n_classes )
        self.n_attribute_classes = 0
        self.attribute_embed_size = 0
        config.n_attributes = 5
        super(Attribute2SequenceModel, self).__init__(helper, config, pretrained_embeddings, cat, test_batch, limit)


