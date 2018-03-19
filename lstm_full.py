#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic LSTM Multiple Attribute Classification
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
import os
import sklearn.metrics
from lstm import RNNModel, Config


def pad_sentences(data, max_length):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.
    
    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word itself a list of
            @n_features features.
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (sentence', labels', mask).
        sentence' and  mask are of length @max_length,
        labels' will be of size len(labels) as these are attributes
        of the entire sentence (i.e. they should remain unchanged)
    """
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * Config.n_features

    for sentence, labels in data:
        ### YOUR CODE HERE (~4-6 lines)
        sentence_copy = sentence[:]
        sentence_length = len(sentence_copy)
        diff = max_length - sentence_length
        if diff >  0:
            sentence_copy += [zero_vector]*diff
        mask = [(i < sentence_length) for i,_ in enumerate(sentence_copy)]
        ret.append((sentence_copy[:max_length], labels , mask[:max_length]))
        ### END YOUR CODE ###
    return ret

class MultiAttributeRNNModel(RNNModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will predict wine attributes (region, points, price, etc.)
    given a twitter review of some wine.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.
        """
        self.input_placeholder = tf.placeholder(tf.int32, shape = (None, self.max_length, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape = (None, self.config.n_attributes, 1))
        self.mask_placeholder = tf.placeholder(tf.bool, shape = (None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape = ())

    def add_attribute_mask(self):
        attribute_mask = tf.constant(self.attribute_mask)
        #self.attribute_mask = tf.Print(self.attribute_mask, [self.attribute_mask])
        attribute_mask = tf.cast(attribute_mask, tf.float32)
        return attribute_mask
        
    def add_prediction_op(self):
        """Adds the unrolled RNN.
        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes, )
        """
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder
        attribute_mask = self.add_attribute_mask()
        #with tf.variable_scope("RNN", reuse = tf.AUTO_REUSE):        
        U = tf.get_variable("OutputWeights", shape = (self.config.hidden_size, self.config.n_classes*self.config.n_attributes), initializer = tf.contrib.layers.xavier_initializer())
        b_2 = tf.get_variable("OutputBias", shape = (self.config.n_classes*self.config.n_attributes), initializer = tf.zeros_initializer())
        
        rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, initializer = tf.contrib.layers.xavier_initializer())
        #init_state = rnn_cell.zero_state(self.config.current_batch_size, dtype = tf.float32) 
        #runs the entire rnn - "state" is the final state of the lstm
        outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                           inputs=x, dtype=tf.float32)
        outputs = tf.nn.dropout(outputs, dropout_rate) 
        #outputs_drop = tf.reduce_mean(outputs, axis = 1)
        mask = tf.cast(self.mask_placeholder, dtype = tf.float32)
        outputs_copy = tf.multiply(outputs, tf.expand_dims(mask, 2)) 
        outputs = tf.reduce_mean(outputs_copy, axis = 1)    
        preds = tf.add(tf.matmul(outputs, U), b_2)
        preds = tf.reshape(preds, [-1, self.config.n_attributes, self.config.n_classes]) 
        preds = tf.multiply(preds, tf.expand_dims(attribute_mask, 0))
        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        labels = self.labels_placeholder
        attribute_mask = self.attribute_mask
        #labels = tf.Print(labels, [labels], message = "True: ", summarize = self.config.max_length) 
        #labels = tf.Print(labels, [tf.shape(labels)], message = "Shape: ")
        labels = tf.Print(labels, [labels], message = "Pre-Loss: ")
        labels = tf.squeeze(labels) 
        labels = tf.Print(labels, [labels], message = "Pre-Loss: ")
        #labels = tf.cast(labels, dtype = tf.int32)   
        #preds = tf.Print(preds, [tf.argmax(tf.nn.softmax(preds), axis = 1)], message = "Pred: ", summarize = self.config.current_batch_size)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = preds)
        labels = tf.Print(labels, [labels], message = "True(reduced): ")
        labels = tf.cast(labels, dtype=tf.bool)
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.multiply(loss, labels)
        loss = tf.reduce_mean(loss)
        return loss

    def preprocess_data(self, data, pad = True):
        '''
        data is of form: [sentence, labels]

        returns: [sentence, labels]
        '''
        data_copy = data
        data_copy = self.data_helper.data_as_list_of_tuples(data_copy)
        if pad:
            data_copy = pad_sentences(data_copy, self.config.max_length)
        return data_copy

    def featurize_data(self, data):
        """featurizes data in specified manner
        """
        pass

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            preds_ = preds[i]
            #preds_ = np.expand_dims(preds[i], axis = 2)
            labels_ = copy.deepcopy(np.squeeze(labels))
            #labels_ = labels[:]
            assert len(preds_) == len(labels_)
            ret.append([sentence, labels_, preds_])
        #print("Predictions (sent, true, pred): ", ret)
        return ret
    

    def evaluate(self, sess, examples_raw, examples = None):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
        sess: the current TensorFlow session.
        examples: A list of vectorized input/output pairs.
        examples_raw: A list of the original input/output sequence pairs.
        Returns:
        The one to one accuracy for predicting tokens as named entities.
        """
        #token_cm = ConfusionMatrix(labels=LBLS)
        def accuracy_score(Y_pred, Y_true, axis = 0):
            '''
            returns: array of accuracy scores of size n_attributes or batch_sze depending on axis
            '''
            accuracy = Y_pred==Y_true
            accuracy = np.mean(accuracy, axis = axis)
            print(accuracy)
            return accuracy
        
        def f1_score(Y_pred, Y_true, average = 'weighted'):
            f1_scores = np.array([])
            for col_pred, col_true in zip(Y_pred.T, Y_true.T):
                np.append(f1_scores, sklearn.metrics.f1_score(col_pred, col_true, average=average))
            return f1_scores

        def test_accuracy(Y_pred,Y_true):
            acc = accuracy_score(Y_pred, Y_true)          
            f1_w  = f1_score(Y_pred,Y_true,average="weighted")  
            f1_m = f1_score(Y_pred,Y_true,average="macro")  
            return acc,f1_w,f1_m

        acc_array = []
        sentences, class_labels, predictions = zip(*self.output(sess, examples_raw, examples))
        print("labels: ",class_labels)
        print("preds: ", predictions)
        labels_np = np.array(class_labels)
        predictions_np = np.array(predictions)
        print("labels_np :", labels_np)
        print("preds_np: ", predictions_np)
        return test_accuracy(predictions_np,labels_np)

        '''
        for _, labels, labels_  in :
            acc_array.append(accuracy_score(labels, labels_))
        one_2_one = np.mean(acc_array)
        return one_2_one
        '''

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
       
        if inputs is None:
            inputs = self.preprocess_data(inputs_raw)
        
        # make copy of raw inputs
        inputs_raw_copy = inputs_raw
        inputs_raw_copy = self.preprocess_data(inputs_raw_copy, pad = False)
         
        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw_copy, inputs, preds)

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        '''
        predicts labels
        '''
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch) 
        predictions = sess.run(self.pred, feed_dict=feed)
        predictions = np.argmax(predictions, axis = 2)
        return predictions
    
    def __init__(self, helper, config, pretrained_embeddings, attributes_mask, cat=None, test_batch=None, limit=None):
        self.attribute_mask = np.array(attributes_mask)
        config.n_attributes = self.attribute_mask.shape[0]
        super(MultiAttributeRNNModel, self).__init__(helper, config, pretrained_embeddings, cat, test_batch, limit)
