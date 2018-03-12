'''
File for accessing and manipulating the data
'''

import pandas as pd
import pprint
import numpy as np
from collections import Counter,defaultdict
import copy,math
import json


train_data_file = "data/train_utf.pkl"  
dev_data_file = "data/dev_utf.pkl"
test_data_file = "data/test_utf.pkl"

X_cat = "description"


class data_helper():

	def __init__(self,max_len=None):
		'''
		Max-len is useful for testing 
		'''
		self.max_length = 0
		self.train_data,self.dev_data,self.test_data = self.load_data(max_len)
		self.X_train = self.train_data[X_cat]
		self.X_dev = self.dev_data[X_cat]
		self.X_test = self.test_data[X_cat]
		self.vocab, self.word_freq_dict = self.generate_vocab_and_word_frequencies()

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

	def get_all_data(self,category):
		#returns the concatenated 
		all_data = np.array([])
		for data in [self.train_data,self.dev_data,self.test_data]:
			all_data = np.append(all_data,data[category])
		return all_data

	def get_Y_cat(self,Y_cat):
		return self.train_data[Y_cat],self.dev_data[Y_cat],self.test_data[Y_cat]

	def discretize(self,category,num_categories=20):
		'''
		Converts a continous distribution into an equally split discrete distribution
		'''
		
		all_data = np.array([])
		for data in [self.train_data,self.dev_data,self.test_data]:
			all_data = np.append(all_data,data[category])
		all_data = all_data.astype(float)
		all_data = np.sort(all_data)
		all_data = all_data[~np.isnan(all_data)]

		discrete_areas = np.array_split(all_data,num_categories)
		bins = [b[0] for b in discrete_areas]
		bins.append(float("inf"))


		ret = []
		for data in [self.train_data,self.dev_data,self.test_data]:
			cat_data = data[category]
			discretized = []
			for y in cat_data:
				if math.isnan(y): 
					discretized.append(None)
				else:
					discretized.append(str(min(set(x for x in bins if x>y))))
			ret.append(discretized)

		return ret 

	def generate_vocab_and_word_frequencies(self):
		'''
		Generates the vocabulary and word frequencies in train, test and dev
		Returns: set of words making up vocab, dictionary of word frequencies {word: count}
		'''
		vocab = set()
		word_counts = defaultdict(int)
		for data in [self.X_train,self.X_dev,self.X_test]:
			for desc in data:
				if len(desc) > self.max_length:
					self.max_length = len(desc)
				for word in desc:
					vocab|=set(word)
					word_counts[word]+=1 
		return vocab, word_counts

if __name__ == "__main__":
	du = data_helper(300)
	freq_dict = du.word_freq_dict
	print (json.dumps(freq_dict, indent=1))
	print(len(du.vocab))
	#print(du.discretize("price",20))