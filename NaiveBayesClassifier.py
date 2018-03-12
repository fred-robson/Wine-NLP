'''
Implements Naive Bayes to run as a baseline for the wine project

'''
from data_utils import data_helper
import numpy as np
from collections import Counter,defaultdict
from sklearn.metrics import confusion_matrix,f1_score
import sys

ALPHA_RANGE = [1,2,3,4,5,10]

def classify(words,doc_counts,word_counts,len_vocab,alpha):
	'''
	Classifies a single example - @words - according to the naive bayes model defined by 
	doc_counts and word_counts

	Return: The best label for the set of words
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
		

def test_accuracy(X,Y,doc_counts,word_counts,len_vocab,alpha,print_iterations):
	'''
	Tests the accuracy of the Naive Bayes model on X,Y 
	
	Returns the accuracy, f1 score and confusion matrix
	'''	
	correct = 0.0 
	incorrect = 0.0 

	Y_pred = []
	Y_true = []

	for i,(x,true_label) in enumerate(zip(X,Y)): 
		pred = classify(x,doc_counts,word_counts,len_vocab,alpha)
		if true_label != None: 
			Y_pred.append(pred)
			Y_true.append(true_label)
			if pred == true_label: correct+=1
			else: incorrect+=1
		if print_iterations:
			if i % 1000 ==0: print("\rIteration ",i,": ",'%.5f'%(correct/(correct+incorrect)),end="\r")
	return float(correct)/(incorrect+correct),f1_score(Y_true,Y_pred,average="macro"),confusion_matrix(Y_true,Y_pred)

def get_best_alpha(X_dev,Y_dev,doc_counts,word_counts,len_vocab,alpha_range,show_iterations):
	'''
	Uses the dev set to get the best alpha 
	'''
	scores = {}
	for a in alpha_range: 
		accuracy,f1,_ = test_accuracy(X_dev,Y_dev,doc_counts,word_counts,len_vocab,a,show_iterations)
		scores[a] = accuracy
		print('\x1b[2K\r',end="\r") #Clears teh line
		print("\rAlpha",a,'%.5f'%accuracy)

	return max(scores,key=scores.get)

def gen_dicts(X_train,Y_train):
	'''
	Converts the sentences into bag of words dicts {word: count}
	'''
	doc_counts = defaultdict(int)
	word_counts = defaultdict(lambda: defaultdict(int))
	for desc,label in zip(X_train,Y_train):
		#skips the blanks
		if type(label)not in [str,int,float]: continue
		doc_counts[label]+=1
		for w in desc: 
			word_counts[label][w]+=1 
	return doc_counts,word_counts


def main(limit,show_iterations):
	du = data_helper(limit)
	len_vocab = len(du.generate_vocab_and_word_frequencies()[0])
	
	all_Y_cats = ["variety","points","price","country","province"]
	
	for Y_cat in all_Y_cats:
		
		if Y_cat == "price":
			Y_train,Y_dev,Y_test =du.discretize(Y_cat)
		else:	
			Y_train,Y_dev,Y_test = du.get_Y_cat(Y_cat)
	
		doc_counts,word_counts = gen_dicts(du.X_train,Y_train)
		print ("Category:",Y_cat)
		print ("------------------------")
		print ("# Categories: ",len(doc_counts))
		print ("Calculating Best Alpha using dev set...")
		alpha = get_best_alpha(du.X_train,Y_dev,doc_counts,word_counts,len_vocab,ALPHA_RANGE,show_iterations)
		print("Best Alpha = ",alpha)
		print ("Calculating Test accuracy")
		accuracy,f1,confusion = test_accuracy(du.X_train,Y_test,doc_counts,word_counts,len_vocab,alpha,show_iterations)
		print ("\rTest Accuracy: ",accuracy)
		print ("F1 Score: ",f1)
		print ("\n\n")

		


if __name__ == "__main__":
	args = sys.argv
	
	limit = None
	if str.isdigit(args[1]): 
		limit = int(args[1])
		print("Limit of ",limit)

	show_iterations = True
	if "-q" in args:
		show_iterations = False
		print("Iterations turned off")
	print ("\n")
	main(limit,show_iterations)


