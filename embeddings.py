'''
File for accessing and manipulating the embeddings
'''

import pandas as pd
import numpy as np
from collections import Counter,defaultdict
import copy,math
import os


glove_embeddings_file_cased = "/embeddings/glove.840B.300d.txt"
glove_embeddings_file_uncased = "/embeddings/glove.42B.300d.txt"


class embedding_helper():

    def __init__(self, vocab = None, embedding_type = "GLOVE_CASED"):
        '''
        instantiates an embedding dictionary 
        '''
        self.embedding_dict, self.embedding_matrix, self.embedding_tokens_vector = self.load_embeddings(
                                                                                vocab, embedding_type)

    def load_embeddings(self, vocab, embedding_type):
        '''
        Loads the embeddings from the .txt file. Called at initialization.
        returns: embeddings
        '''
        loaded_embeddings = []
        embeddings = None
        cwd = os.getcwd()
        if embedding_type == "GLOVE_CASED":
            embeddings = self.load_glove_embeddings(cwd+glove_embeddings_file_cased)
        elif embedding_type == "GLOVE_UNCASED":
            embeddings = self.load_glove_embeddings(cwd+glove_embeddings_file_uncased)
        else:
            print ("No existing embeddings: ", embedding_type)
        if vocab is not None: embeddings = {k:v for k, v in embeddings.items() if k in vocab}
        return embeddings 
    
    @staticmethod
    def load_glove_embeddings(gloveFile):
        print ("Loading Glove Embeddings: ", gloveFile)
        try:
            f = open(gloveFile,'r')
        except:
            print("File Error: Can't find file ", gloveFile)
        embeddings_dict = {}
        embedding_matrix = []
        embedding_tokens_vector = []
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            try:
                weights = np.array([float(val) for val in splitLine[1:]])
            except:
                print("Word: ", word)
                print("weights: ", splitLine[1:])
            embeddings_dict[word] = weights
            embedding_matrix.append(weights)
            embedding_tokens_vector.append(word)
        print ("Done.",len(embeddings)," words loaded!")
        return embeddings_dict, embedding_matrix, embedding_tokens_vector

    def get_embedding_weights(self, word):
        '''
        checks to see if there are embeddings for the desired word 
        returns: weights
        '''
        assert self.embedding_dict is not None, ("No embeddings have been loaded")
        weights = self.embedding_dict.get(word, "")
        assert weights is not "", ("No embeddings for ", word)        
        return weights
    
    def get_embedding_matrix(self, vocab = None):
        '''
        Checks to see if there are embeddings for the entire vocab
        if a vocab is given, otherwise gets the entire embedding matrix
        returns: embedding weights matrix
        '''
        assert self.embedding_matrix is not None, ("No embeddings have been loaded")
        if vocab is not None:
            return self.embedding_matrix
        
    def get_tokens_as_embedding_indices(self, tokens):
        '''
        There must be embeddings for all tokens given.
        returns: returns embedding indices of given tokens
        '''
        assert self.embedding_tokens_vector is not None, ("No embeddings have been loaded")
        indices = np.arange(len(self.embedding_tokens_vector))
        mask = self.embedding_tokens_vector[tokens]
        indices = indices[mask]
        return indices
        


if __name__ == "__main__":
    eu = embedding_helper()
    eu.load_embeddings()
    print(eu.get_embedding_weights("word"))