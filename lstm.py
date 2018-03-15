#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic LSTM
"""
import argparse
import sys
import time
from datetime import datetime
from util import Progbar

import tensorflow as tf
import numpy as np
import copy
from model import Model
from util import minibatches
import os
from sklearn.metrics import accuracy_score

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_features = 1 # Number of features for every word in the input.
    max_length = 120 # longest sequence to parse
    n_classes = 5
    n_attributes = 1
    dropout = 0.5
    embed_size = 50
    hidden_size = 100
    batch_size = 32
    current_batch_size = batch_size
    n_epochs = 30
    max_grad_norm = 10.
    lr = 0.001

    def __init__(self, cell, n_classes = 0, many2one = False):
        self.cell = cell
        self.many2one = many2one
        if n_classes:
            self.n_classes = n_classes
        self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

def pad_sequences(data, max_length, many2one = False):
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

    for sentence, labels in data:
        ### YOUR CODE HERE (~4-6 lines)
        labels_copy = labels[:]
        sentence_copy = sentence[:]
        sentence_length = len(sentence_copy)
        diff = max_length - sentence_length
        if diff >  0:
            sentence_copy += [zero_vector]*diff
            labels_copy += [zero_label]*diff
        mask = [(i < sentence_length) for i,_ in enumerate(sentence_copy)]
        ret.append((sentence_copy[:max_length], labels_copy[:max_length] , mask[:max_length]))
        ### END YOUR CODE ###
    return ret

class RNNModel(Model):
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
        self.labels_placeholder = tf.placeholder(tf.int32, shape = (None, self.max_length))
        self.mask_placeholder = tf.placeholder(tf.bool, shape = (None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape = ())
        
    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for training model.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout, self.mask_placeholder: mask_batch}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        embeddings = tf.get_variable("embeddings", initializer = self.pretrained_embeddings)
        embeddings = tf.nn.embedding_lookup(embeddings, self.input_placeholder)
        embeddings = tf.reshape(embeddings, [-1, self.max_length, self.config.n_features* self.config.embed_size])
        embeddings = tf.cast(embeddings,tf.float32) 
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
        
        rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, initializer = tf.contrib.layers.xavier_initializer())
        #init_state = rnn_cell.zero_state(self.config.current_batch_size, dtype = tf.float32) 
        #runs the entire rnn - "state" is the final state of the lstm
        outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                           inputs=x, dtype=tf.float32)
        outputs = tf.nn.dropout(outputs, dropout_rate) 
        #outputs_drop = tf.reduce_mean(outputs, axis = 1)
        if self.config.many2one:
            mask = tf.cast(self.mask_placeholder, dtype = tf.float32)
            outputs = tf.multiply(outputs, tf.expand_dims(mask, 2))
            outputs = tf.reduce_mean(outputs, axis = 1)
        else:
            outputs = tf.reshape(outputs, [-1, self.config.hidden_size])
        preds = tf.add(tf.matmul(outputs, U), b_2)
        if not self.config.many2one:
            preds = tf.reshape(preds, [-1, self.config.max_length, self.config.n_classes]) 
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
        if self.config.many2one:
            labels = tf.reduce_mean(labels, axis = 1)
            print(labels) 
            labels = tf.cast(labels, dtype = tf.int32)
        else:
            preds = tf.boolean_mask(preds, self.mask_placeholder)
            labels = tf.boolean_mask(self.labels_placeholder, self.mask_placeholder)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = preds)
        loss = tf.reduce_mean(loss)
        return loss, labels

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate = self.config.lr)
        train_op = optimizer.minimize(loss) 
        return train_op
    
    def preprocess_data(self, data, pad = True):
        '''
        data is of form: [sentence, labels]

        returns: [sentence, labels]
        '''
        #make copy of data
        data_copy = data
        data_copy = self.data_helper.data_as_list_of_tuples(data_copy)
        data_copy = self.format_labels(data_copy)
        if pad:
            data_copy = pad_sequences(data_copy, self.config.max_length)
        return data_copy

    def format_labels(self, data):
        """
        makes sure labels  are of same dims as 
        corresponding placeholders

        data is a list of tuples: (sentence, labels)

        returns: [(sentence, labels),... ] (correctly formatted)
        """
        ret = []
        for sentence, labels in data:
            assert len(labels) == self.config.n_attributes, ("Invalid number of labels for given sentence")

            labels_copy = copy.deepcopy(labels)
            sentence_length = len(sentence)
            diff = sentence_length - len(labels_copy)
            if diff > 0:
                labels_copy += labels_copy*diff
            ret.append((sentence, labels_copy))
        return ret

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
            _, _, mask = examples[i]
            labels_ = None
            labels_gt = labels[:]
            if self.config.many2one:
                labels_ = [preds[i]]
                labels_gt = [labels[0]]
            else:
                labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            assert len(labels_) == len(labels_gt)
            ret.append([sentence, labels_gt, labels_])
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
        acc_array = []
        for _, labels, labels_  in self.output(sess, examples_raw, examples):
            acc_array.append(accuracy_score(labels, labels_))
        one_2_one = np.mean(acc_array)
        return one_2_one

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
        '''def preds_many2one(predictions):
            int_mask = tf.cast(mask_batch, dtype = tf.float32)
            predictions = tf.multiply(predictions, tf.expand_dims(int_mask, 2))
            predictions = tf.nn.softmax(predictions)
            predictions = tf.reduce_sum(predictions, axis = 1)
            predictions = tf.argmax(predictions, axis = 1)
            return predictions
        '''
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        axis = 2
        if self.config.many2one:
            axis = 1
        predictions = sess.run(self.pred, feed_dict=feed)
        predictions = np.argmax(predictions, axis = axis)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        self.config.current_batch_size = inputs_batch.shape[0]
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=Config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss
    
    def fit(self, sess, saver, train_raw, dev_set_raw):
        train = self.preprocess_data(train_raw)
        #dev_set = self.preprocess_data(dev_set_raw)
        best_score = 0.
        for epoch in range(self.config.n_epochs):
            print("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
            loss = []
            for minibatch in minibatches(train, self.config.batch_size):
                loss.append([self.train_on_batch(sess, *minibatch)])
            loss = np.array(loss)
            loss = np.mean(loss)
            print("Loss: ", loss)
            print("")
            #print(dev_set_raw)
            score_train  = self.evaluate(sess, train_raw)
            score = self.evaluate(sess, dev_set_raw)
            #score = 0.0
            print("Accuracy | Train: %f , Dev: %f", (score_train, score))
            if score > best_score:
                best_score = score
                if saver:
                    print("New best accuracy! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
        return best_score
    
    def __init__(self, helper, config, pretrained_embeddings):
        self.data_helper = helper
        self.config = config
        self.max_length = min(Config.max_length, helper.max_length)
        Config.max_length = self.max_length
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()
