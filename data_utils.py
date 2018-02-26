'''
File for accessing and manipulating the data
'''

import pandas as pd
import numpy as np
from collections import Counter,defaultdict


train_data_file = "data/train_utf.pkl"  
dev_data_file = "data/dev_utf.pkl"
test_data_file = "data/test_utf.pkl"

X_cat = "description"


class data_utils():

	def __init__(self,max_len=None):
		self.train_data,self.dev_data,self.test_data = self.load_data(max_len)
		self.X_train = self.train_data[X_cat]
		self.X_dev = self.dev_data[X_cat]
		self.X_test = self.test_data[X_cat]

	def load_data(self,max_len=None):
		'''
		Loads the data from the pickle file. Called at initialization 
		returns: train_data,dev_data,test_data
		'''
		loaded_data = []
		for filename in [train_data_file,dev_data_file,test_data_file]:
			data_frame = pd.read_pickle(filename)
			if max_len == None: loaded_data.append(data_frame)
			else: loaded_data.append(data_frame[:max_len])
		return loaded_data

	def gen_dicts(self,X_train,Y_train):
		doc_counts = defaultdict(int)
		word_counts = defaultdict(lambda: defaultdict(int))
		for desc,label in zip(X_train,Y_train):
			#skips the blanks
			if type(label)!=str: continue
			doc_counts[label]+=1
			for w in desc: 
				word_counts[label][w]+=1 
		return doc_counts,word_counts

	def get_len_vocab(self):
		'''
		Gets the length of the vocabulary contrained in train, test and dev
		Returns: int lenght of vocab
		'''
		Vocab = set()
		for data in [self.X_train,self.X_dev,self.X_test]:
			for word in data:
				Vocab|=set(word)
		return len(Vocab)	


if __name__ == "__main__":
	pass