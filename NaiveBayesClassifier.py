#Runs Naive Bayes as a baseline
import pandas as pd
import numpy as np
from collections import Counter,defaultdict
from tqdm import tqdm

train_data_file = "./data/train_utf.pkl"  
dev_data_file = "./data/dev_utf.pkl"
test_data_file = "./data/test_utf.pkl"
X_cat = "description"
ALPHA_RANGE = [1,2,5,10]


def load_data(max_len=None):
	'''
	Loads the data from the pickle file
	returns: train_data,dev_data,test_data
	'''
	loaded_data = []
	for filename in [train_data_file,dev_data_file,test_data_file]:
		data_frame = pd.read_pickle(filename)
		if max_len == None: loaded_data.append(data_frame)
		else: loaded_data.append(data_frame[:max_len])
	return loaded_data

def gen_dicts(X_train,Y_train):
	doc_counts = defaultdict(int)
	word_counts = defaultdict(lambda: defaultdict(int))
	for desc,label in zip(X_train,Y_train):
		#skips the blanks
		if type(label)!=str: continue
		doc_counts[label]+=1
		for w in desc: 
			word_counts[label][w]+=1 
	return doc_counts,word_counts

def classify(words,doc_counts,word_counts,len_vocab,alpha):
	'''
	Classifies a single example - @words - according to the naive bayes model defined by 
	doc_counts and word_counts
	'''

	doc_counts_sum = float(sum(doc_counts.values()))
	all_LLs = {}

	for label in doc_counts: 

		#Get likelihood of document
		log_likelihood = np.log(doc_counts[label]/doc_counts_sum)	
		
		word_counts_sum = sum(word_counts[label].values())
		word_counts_sum += alpha*len_vocab
		#Avoid integer division
		word_counts_sum = float(word_counts_sum)

		for w in words: 
			w_count = word_counts[label][w]+alpha
			log_likelihood += np.log(w_count/word_counts_sum)

		all_LLs[label] = log_likelihood

	return max(all_LLs,key=all_LLs.get)

def get_len_vocab(train,dev,test):
	Vocab = set()
	for data in [train,dev,test]:
		for index, row in data.iterrows():
				Vocab|=set(row[X_cat])
	return len(Vocab)			

def test_accuracy(X_test,Y_test,doc_counts,word_counts,len_vocab,alpha):
	'''
	Tests the accuracy of the Naive Bayes model on X_test,Y_test 
	'''	
	correct = 0.0 
	incorrect = 0.0 
	for i,(x,true_label) in enumerate(zip(X_test,Y_test)): 
		pred = classify(x,doc_counts,word_counts,len_vocab,alpha)
		if pred == true_label: correct+=1
		else: incorrect+=1
		if i % 1000 ==0: print("\r Iteration ",i,": ",correct/(correct+incorrect),end=".")
	return float(correct)/(incorrect+correct)

def get_best_alpha(X_dev,Y_dev,doc_counts,word_counts,len_vocab,alpha_range):
	'''
	Uses the dev set to get the best alpha 
	'''
	scores = {}
	for a in alpha_range: 
		accuracy = test_accuracy(X_dev,Y_dev,doc_counts,word_counts,len_vocab,a)
		scores[a] = accuracy
		print("\rAlpha",a,accuracy)
	return max(scores,key=scores.get)


def main():
	train_data,dev_data,test_data = load_data()
	len_vocab = get_len_vocab(train_data,dev_data,test_data)
	
	all_Y_cats = ["country","variety","province"]
	
	for Y_cat in all_Y_cats:
		X_train = train_data[X_cat]
		Y_train = train_data[Y_cat]		
		doc_counts,word_counts = gen_dicts(X_train,Y_train)
		print ("Category:",Y_cat)
		print ("------------------------")
		print ("# Categories: ",len(doc_counts))
		print ("Calculating Best Alpha using dev set...")
		alpha = get_best_alpha(dev_data[X_cat],dev_data[Y_cat],doc_counts,word_counts,len_vocab,ALPHA_RANGE)
		print ("Calculating Test accuracy")
		accuracy = test_accuracy(dev_data[X_cat],dev_data[Y_cat],doc_counts,word_counts,len_vocab,alpha)
		print ("Test Accuracy: ",accuracy)
		print ("\n\n")

		


if __name__ == "__main__":
	main()