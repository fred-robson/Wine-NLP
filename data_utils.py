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
		self.vocab_to_index = {v:i for i,v in enumerate(self.vocab)}

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

	def get_Y_cat(self,Y_cat,normalize=False):
		#TO DO: Add normalization
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

	def missing_indices(self,Y_cat):
		'''
		Returns a vector of indices where the Y category is empty
		'''	
		ret = []
		Y_data = self.get_Y_cat(Y_cat)
		for Y in Y_data:
			indices = []
			for i,y_i in enumerate(Y):
				if not type(y_i) is str and math.isnan(y_i): 
					indices.append(i)
			ret.append(indices)
		return ret 

	def filtered_on_missing_indices(self,Y_cat):
		#Removes anything that has a missing Y value
		X = self.X_train, self.X_dev, self.X_test
		Y = self.get_Y_cat(Y_cat)
		Z = self.missing_indices(Y_cat)
	
		ret = [],[]

		for x,y,z in zip(X,Y,Z):
			ret[0].append(np.delete(np.array(x),z))
			ret[1].append(np.delete(np.array(y),z))
		return ret
	

	def get_vectorized_X(self,all_X=None):
		'''
		Converts the ["I","am"...."am"] into a vector where the index represents a word count
		'''
		ret = []
		if all_X is None: 
			all_X =[self.X_train,self.X_dev,self.X_test] 
			
		for X  in all_X:
			X_vectorized = np.zeros([len(X),len(self.vocab)])
			for i,row in enumerate(X):
				for w in row:
					j = self.vocab_to_index[w]
					X_vectorized[(i,j)]+=1
			ret.append(X_vectorized)
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
					vocab.add(word)
					word_counts[word]+=1 
		return vocab, word_counts

if __name__ == "__main__":
	du = data_helper(1000)
	X,Y = du.filtered_on_missing_indices("province")
	print (len(X[0]))
	
	#print(du.discretize("price",20))