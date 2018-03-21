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
from lstm import RNNModel, Config

def softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
    # Matrix
        c = np.amax(x, axis = -1)
        c = np.reshape(c, (c.shape[0], c.shape[1], 1))
        num = np.exp(x - c)
        den = np.sum(num, axis = -1)
        x = num/(np.reshape(den, (den.shape[0], den.shape[1], -1)))
    else:
        # Vector
        c = np.amax(x)
        x = np.exp(x - c)/np.sum(np.exp(x-c))
    assert x.shape == orig_shape
    return x

def pad_data(data, max_length):
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
    zero_label = 0

    for sentence, labels, attr in data:
        ### YOUR CODE HERE (~4-6 lines)
        labels_copy = labels[:]
        sentence_copy = sentence[:]
        sentence_length = len(sentence_copy)
        diff = max_length - sentence_length
        if diff >  0:
            sentence_copy += [zero_vector]*diff
            labels_copy += [zero_label]*diff
        mask = [(i < sentence_length) for i,_ in enumerate(sentence_copy)]
        ret.append((sentence_copy[:max_length], labels_copy[:max_length] , mask[:max_length], attr))
        ### END YOUR CODE ###
    return ret

class Attribute2SequenceModel(RNNModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will generate wine reviews given attributes (region, points, price, etc.)
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.
        """
        self.input_placeholder = tf.placeholder(tf.int32, shape = (self.config.batch_size, None, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape = (self.config.batch_size, None))
        self.mask_placeholder = tf.placeholder(tf.bool, shape = (self.config.batch_size, None))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape = ())
        self.attribute_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.n_attributes))
        self.init_state_placeholder = tf.placeholder(tf.float32, [2, 2, None, self.config.hidden_size])

    def add_attribute_embedding(self):
        embeddings = tf.get_variable("attribute_embeddings", shape=(self.n_attribute_classes, self.attribute_embed_size),initializer = tf.contrib.layers.xavier_initializer())
        embeddings = tf.nn.embedding_lookup(embeddings, self.attribute_placeholder)
        embeddings = tf.reshape(embeddings, [-1, self.config.hidden_size, self.config.n_attributes])
        embeddings = tf.cast(embeddings, tf.float32)
        embeddings = tf.reduce_sum(embeddings, -1)

        state_per_layer_list = tf.unstack(self.init_state_placeholder, axis=0)

        embeddings = tuple([tf.nn.rnn_cell.LSTMStateTuple(tf.divide(tf.add(embeddings, state_per_layer_list[i][0]), 2.0),tf.divide(tf.add( embeddings, state_per_layer_list[i][1]),2.0 )) for i in range(self.config.n_layers)])
        return embeddings

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        #with tf.variable_scope("RNN", reuse = tf.AUTO_REUSE):
        embeddings = tf.get_variable("embeddings", initializer = self.pretrained_embeddings,trainable=True)
        inputs = self.input_placeholder
        inputs = tf.reshape(inputs, [self.config.batch_size, -1 , self.config.n_features])
        embeddings = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        embeddings = tf.reshape(embeddings, [self.config.batch_size, -1, self.config.n_features* self.config.embed_size])
        embeddings = tf.cast(embeddings, tf.float32)
        return embeddings

    def add_outputs_op(self):
        x = self.add_embedding() 

        #last_state = tf.cond(self.test_time, lambda:self.add_output_state() ,lambda: self.add_attribute_embedding())
        
        last_state = self.add_attribute_embedding()
        rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, initializer = tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
        #runs the entire rnn - "state" is the final state of the lstm
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell]*self.config.n_layers, state_is_tuple=True) 
        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=x, dtype=tf.float32, initial_state = last_state)
        return outputs, state


    def add_prediction_op(self, outputs):
        """Adds the unrolled RNN.
        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        dropout_rate = self.dropout_placeholder
        U = tf.get_variable("OutputWeights", shape = (self.config.hidden_size, self.config.n_classes), initializer = tf.contrib.layers.xavier_initializer())
        b_2 = tf.get_variable("OutputBias", shape = (self.config.n_classes), initializer = tf.zeros_initializer())

        outputs = tf.nn.dropout(outputs, dropout_rate) 

        outputs = tf.reshape(outputs, [-1, self.config.hidden_size])    
        preds = tf.add(tf.matmul(outputs, U), b_2)
        preds = tf.reshape(preds, [self.config.batch_size, -1, self.config.n_classes])
        #preds = tf.Print(preds, [preds], summarize = self.config.n_classes)
        return preds 

    def create_feed_dict(self, inputs_batch, attributes, mask_batch=None, labels_batch=None, dropout=1, init_state=None):
        """Creates the feed_dict for training model.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        if init_state is None:
            init_state = np.zeros((self.config.n_layers, 2, self.config.batch_size, self.config.hidden_size))
        feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout,  self.attribute_placeholder: attributes, self.init_state_placeholder: init_state}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if mask_batch is not None:
            feed_dict[self.mask_placeholder]=mask_batch
        return feed_dict


    def generate(self, sess, attributes):
        def preds_over_batch(preds):
            guesses = np.array([])
            for i,pred in enumerate(preds):
                dist = np.squeeze(pred)
                guess = np.random.choice(dist.shape[0],1,p= dist)
                guesses = np.append(guesses, guess)
            return guesses

        
        start_data = np.array([self.start_ind])
        start_data = np.expand_dims(start_data, 0)
        start_data_batch = [start_data for _ in range(self.config.batch_size)]
        start_data = np.stack(start_data_batch, axis=0)
        feed = self.create_feed_dict(inputs_batch=start_data, attributes=attributes)

        sentence=start_data

        for i in range(self.config.max_length):
            current_state,preds = sess.run([self.current_state,self.pred], feed_dict=feed)
            preds = softmax(preds)
            preds = preds_over_batch(preds)
            preds = np.expand_dims(preds, -1)
            preds = np.expand_dims(preds, -1)
            sentence = np.append(sentence, preds, axis=1)
            preds = np.copy(preds)
            feed = self.create_feed_dict(inputs_batch=preds, attributes=attributes, init_state=current_state)
        return sentence

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
                pred = np.array(pred)
                accuracy = np.array(np.argmax(pred, axis=1))==np.array(true)
                acc_array = np.append(acc_array,np.mean(accuracy))
            return np.mean(acc_array)
        
        def perplexity(Y_pred, Y_true):
            PP = np.array([])
            for pred, true in zip(Y_pred, Y_true):
                pred = np.array(pred)
                sentence_length = pred.shape[0]
                #print(pred.shape)
                true = np.array(true)
                probs = pred[np.arange(0,true.shape[0]), true]
                #print(probs.shape)
                #print(probs)
                #exit()
                probs_inv = 1.0/probs
                probs_inv = np.log(probs_inv)
                prob_inv_sum = np.sum(probs_inv)/sentence_length
                PP = np.append(PP, np.exp(prob_inv_sum))
            return np.mean(PP)

        def bleu_score(Y_pred=None, Y_true=None):
            return 0

        def test_accuracy(Y_pred,Y_true):
            acc = np.mean(accuracy_score(Y_pred, Y_true))
            PP = perplexity(Y_pred, Y_true)
            bleu = bleu_score()
            #f1_w  = np.mean(f1_score(Y_pred,Y_true,average="weighted"))  
            #f1_m = np.mean(f1_score(Y_pred,Y_true,average="macro"))  
            return acc,PP,bleu

        acc_array = []
        sentences, class_labels, predictions, attr = zip(*self.output(sess, examples_raw, examples))
        return test_accuracy(predictions,class_labels)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels, attributes) in enumerate(examples_raw):
            _, _, mask, _ = examples[i]
            labels_ = None
            #print("labels:", labels)
            #print("preds unmasked:", preds[i])
            labels_gt = labels[:]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            #print("preds:", labels_)
            assert len(labels_) == len(labels_gt)
            #print("labels np:", np.array(labels))
            #print(" ")
            ret.append([sentence, labels_gt, labels_, attributes])
        #print("Predictions (sent, true, pred): ", ret)
        return ret

    def report_results(self, sess, saver, result_train, result_dev, result_test, best_dev_result, train_result_best, best_epoch, epoch, loss):
        
        print("     | Acc      PP      BLEU   |")
        print("     |-------------------------|")
        print("Train| %.3f    %.3f    %.3f |"%(result_train[0],result_train[1],result_train[2]))
        print(" Dev | %.3f    %.3f    %.3f |"%(result_dev[0],result_dev[1],result_dev[2]))
        print("     |-------------------------|\n")

        if result_dev[self.config.result_index] > best_dev_result[self.config.result_index]:
            best_dev_result = result_dev
            train_result_best = result_train
            best_epoch = epoch
            if saver:
                print("New best accuracy! Saving model in %s"%self.config.model_output)
                saver.save(sess, self.config.model_output)
                self.save_model_description()

        self.save_epoch_outputs(epoch,loss,result_dev,result_train,result_test)
        return best_dev_result, train_result_best, best_epoch


    def predict_on_batch(self, sess, inputs_batch, mask_batch, attribute_batch):
        '''def preds_many2one(predictions):
            int_mask = tf.cast(mask_batch, dtype = tf.float32)
            predictions = tf.multiply(predictions, tf.expand_dims(int_mask, 2))
            predictions = tf.nn.softmax(predictions)
            predictions = tf.reduce_sum(predictions, axis = 1)
            predictions = tf.argmax(predictions, axis = 1)
            return predictions
        '''
        feed = self.create_feed_dict(inputs_batch=inputs_batch, attributes=attribute_batch, mask_batch=mask_batch)
        predictions = sess.run(self.pred, feed_dict=feed)
        predictions = softmax(predictions)
        return predictions
    
    
    def train_on_batch(self, sess,  inputs_batch, labels_batch, mask_batch, attributes):
        self.config.current_batch_size = inputs_batch.shape[0]
        feed = self.create_feed_dict(inputs_batch, attributes,labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=self.config.dropout)
        #summary, _, loss = sess.run([merged_summaries, self.train_op, self.loss], feed_dict=feed)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    
    def preprocess_data(self, data, pad = True):
        '''
        data is of form: [sentence, labels]

        returns: [sentence, labels]
        '''
        data_copy = data
        data_copy = self.data_helper.data_as_list_of_tuples(data_copy)
        data_copy = self.format_labels(data_copy)
        if pad:
            data_copy = pad_data(data_copy, self.config.max_length)
        return data_copy

    
    def format_labels(self, data):
        """
        makes sure labels  are of same dims as 
        corresponding placeholders

        data is a list of tuples: (sentence, labels)

        returns: [(sentence, labels),... ] (correctly formatted)
        """
        ret = []
        for sentence, labels, attr in data:
            sentence_length = len(sentence)
            labels_copy = copy.deepcopy(labels)
            labels_copy = [label[0] for label in labels_copy if type(label) is list ]
            ret.append((sentence, labels_copy, attr))
        return ret

    def build(self):
        self.add_placeholders()
        self.outputs, self.current_state = self.add_outputs_op()
        self.pred = self.add_prediction_op(self.outputs)
        #self.pred_test = self.add_prediction_op(raw = True)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)


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

    def save_epoch_outputs(self,epoch,loss,result_dev,result_train, result_test):
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
                f.write("Loss,dev_ACC,dev_PP,dev_BLEU,train_ACC,train_PP,train_BLEU,test_ACC,test_PP,testBLEU,epoch\n")

        with open(self.config.epochs_csv,"a") as f:
            f.write(str(loss)+",")
            for r in result_dev:f.write(str(r)+",") 
            for r in result_train:f.write(str(r)+",") 
            for r in result_test: f.write(str(r)+",")
            f.write(str(epoch)+"\n")



    def __init__(self, helper, config, pretrained_embeddings,cat=None,test_batch=None,limit=None, many2one=False, start_ind = None, end_ind = None):
        print("Num Classes: ",config.n_classes )
        self.n_attribute_classes = helper.n_attribute_classes
        self.attribute_embed_size = config.hidden_size
        config.n_attributes = 4
        self.start_ind = start_ind
        self.end_ind = end_ind
        super(Attribute2SequenceModel, self).__init__(helper, config, pretrained_embeddings, cat, test_batch, limit)


