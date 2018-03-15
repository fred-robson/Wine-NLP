'''
File for accessing and manipulating the embeddings
'''

import pandas as pd
import numpy as np
from collections import Counter,defaultdict
import copy,math
import os
from tqdm import tqdm
import pickle

glove_embeddings_file_cased = "/embeddings/glove.840B.300d.txt"
glove_embeddings_file_uncased = "/embeddings/glove.42B.300d.txt"

class embedding_helper():

    def __init__(self, save_to_pickle = False, from_pickle = False, vocab = None, embedding_type = "GLOVE_UNCASED", test_batch = 0):
        '''
        instantiates an embedding helper 
        '''
        self.embedding_dict, self.embedding_matrix, self.embedding_tokens_2_ind, self.ind_2_embedding_tokens  =  None, None, None, None
        self.unk_indice = 0
        self.unknown_token = "<unk>"
        self.test_batch = test_batch
        if from_pickle:
            self.embedding_dict, self.embedding_matrix, self.embedding_tokens_2_ind, self.ind_2_embedding_tokens = self.load_from_pickle(vocab, embedding_type)
        else:
            self.embedding_dict, self.embedding_matrix, self.embedding_tokens_2_ind, self.ind_2_embedding_tokens = self.load_embeddings(vocab, embedding_type)
            if save_to_pickle:
                self.save_as_pickle("wine_embeddings_"+embedding_type+".pkl")
                
            

    def load_embeddings(self, vocab = None, embedding_type = "GLOVE_UNCASED"):
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
    
    def load_glove_embeddings(self, gloveFile):
        print ("Loading Glove Embeddings: ", gloveFile)
        try:
            f = open(gloveFile,'r')
        except:
            print("File Error: Can't find file ", gloveFile)
        embeddings_dict = {}
        embedding_matrix = []
        embedding_tokens_2_ind = {}
        ind_2_embedding_tokens = {}
        for ind, line in tqdm(enumerate(f)):
            splitLine = line.split()
            word = splitLine[0]
            try:
                weights = np.array([float(val) for val in splitLine[1:]])
            except:
                pass
                #print("Word: ", word)
                #print("weights: ", splitLine[1:])
            embeddings_dict[word] = weights
            embedding_matrix.append(weights)
            embedding_tokens_2_ind[word] = ind
            ind_2_embedding_tokens[ind] = word 
            if self.test_batch > 0:
                if ind == self.test_batch:
                    break
        self.unk_indice = len(embedding_matrix)
        unk_weights = np.zeros_like(embedding_matrix[0])
        embeddings_dict[self.unknown_token] = unk_weights
        embedding_matrix.append(unk_weights)
        embedding_tokens_2_ind[self.unknown_token] = self.unk_indice
        ind_2_embedding_tokens[self.unk_indice] = self.unknown_token 
        print ("Done.",len(embedding_matrix)," words loaded!")
        return embeddings_dict, embedding_matrix, embedding_tokens_2_ind, ind_2_embedding_tokens

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
        return self.embedding_matrix
        
    def get_tokens_as_embedding_indices(self, seq, lookup_dict = None, unk_indice = None):
        '''
        Missing embeddings are given indice corresponding to <unk> token
        returns: returns embedding indices of given tokens
        '''
        tok2ind = False
        if type(seq[0]) is str:
            tok2ind = True
        if lookup_dict is None:
            lookup_dict = self.embedding_tokens_2_ind
        if unk_indice is None:
            unk_indice =  self.unk_indice
        if tok2ind:
            return [[lookup_dict.get(tok.lower(), unk_indice)] for tok in seq]
        else:
            return [lookup_dict.get(ind[0], self.unknown_token) for ind in seq]

    def tok2ind_ind2tok(self, data, lookup_dict = None, unk_indice = None):
        return[self.get_tokens_as_embedding_indices(desc, lookup_dict = lookup_dict, unk_indice = unk_indice) for desc in data]


    def get_sub_embeddings(self, vocab):
        '''
        Missing embeddings are given indice corresponding to <unk> token
        returns: returns embedding_matrix of size len(vocab), corresponding
        embedding_tokens_2_ind, and the index of the <unk> token
        (these are different from the full embedding matrix, dictionary, and
        unknown token index)
        '''
        sub_embedding_matrix = []
        sub_embedding_tokens_2_ind = {}
        sub_ind_2_embedding_tokens = {}
        sub_ind = 0
        for tok in tqdm(vocab):
            ind = self.embedding_tokens_2_ind.get(tok.lower(), self.unk_indice)
            if ind != self.unk_indice:
                sub_embedding_matrix.append(self.embedding_matrix[ind])
                sub_embedding_tokens_2_ind[tok] = sub_ind
                sub_ind_2_embedding_tokens[sub_ind] = tok
                sub_ind += 1
        sub_unk_weights = np.zeros_like(sub_embedding_matrix[0])
        sub_unk_indice = len(sub_embedding_matrix)
        sub_embedding_matrix.append(sub_unk_weights)
        sub_embedding_tokens_2_ind[self.unknown_token] = sub_unk_indice
        sub_ind_2_embedding_tokens[sub_unk_indice] = self.unknown_token
        print ("Done.",len(sub_embedding_matrix)," words loaded!")
        return sub_embedding_matrix, sub_embedding_tokens_2_ind, sub_ind_2_embedding_tokens, sub_unk_indice
    
    def save_as_pickle(self, filename):
        '''
        Saves embeddings as pickle file in the following format:
        [embedding_dict,  embedding_matrix, embedding_tokens_vector]
        returns: nothing
        '''
        cwd = os.getcwd()
        path = cwd+"/pickles/"+filename
        list_to_save = [self.embedding_dict, self.embedding_matrix, self.embedding_tokens_2_ind]
        with open(path, 'wb') as f:
            pickle.dump(list_to_save, f)

    def load_from_pickle(self, filename):
        '''
        Loads embeddings from pickle file in the following format:
        [embedding_dict,  embedding_matrix, embedding_tokens_vector]
        returns: [embedding_dict,  embedding_matrix, embedding_tokens_vector]
        '''
        list_to_return = None
        cwd = os.getcwd()
        path = cwd+"/pickles/"+filename
        with open(path, 'rb') as f:
            list_to_return = pickle.load(f)
        return list_to_return
        
        


if __name__ == "__main__":
    eu = embedding_helper(test_batch = 100)
    print(eu.get_tokens_as_embedding_indices([".", ","]))
    _, t2i, unk_ind = eu.get_sub_embeddings([".", ","])
    print(eu.get_tokens_as_embedding_indices([".", ","],tok_2_ind = t2i, unk_indice = unk_ind))
