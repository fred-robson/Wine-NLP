'''
Implements Naive Bayes to run as a baseline for the wine project

'''
from data_utils import data_helper
import numpy as np
from collections import Counter,defaultdict
import sklearn.metrics
from sklearn import naive_bayes 
import sys

ALPHA_RANGE = [1,2,3,4,5,10]


def test_accuracy(Y_pred,Y_true):
	acc = sklearn.metrics.accuracy_score(Y_pred,Y_true)
	f1  = sklearn.metrics.f1_score(Y_pred,Y_true,average="macro")	
	return acc,f1

def run_NB(alpha,X,Y,NB,test_index = 1):
	#test_index = 1 assumes that testing on dev set not test set 
	naive_bayes = NB(alpha)
	naive_bayes.fit(X[0],Y[0])
	Y_pred = naive_bayes.predict(X[1])
	Y_true = Y[1]
	return test_accuracy(Y_pred,Y_true)

def run_for_Y_cat(X,Y,Y_cat,NB,alpha_range):
	'''
	Assumes that X and Y are a tuple of three 
	'''
	print ("Category:",Y_cat)
	print ("------------------------")
	#print ("# Categories: ",len(doc_counts))
	print ("Calculating Best Alpha using dev set...")
	alpha_scores = {}
	for alpha in alpha_range:
		acc,f1 = run_NB(alpha,X,Y,NB)
		alpha_scores[alpha] = f1
		print("Alpha ",alpha,": Acc = ",acc,", F1 = ",f1)
	best_alpha = max(alpha_scores,key=alpha_scores.get)
	print("Best Alpha = ",best_alpha)
	print ("Calculating Test accuracy")
	acc,f1 = run_NB(best_alpha,X,Y,NB,2)
	print ("\rTest Accuracy: ",acc)
	print ("F1 Score: ",f1)
	print ("\n\n")



def main(limit):
	du = data_helper(limit)
	
	discrete_Y_cats = ["province","variety","country"]
	cont_Y_cats = ["price","points"]

	for Y_cat in discrete_Y_cats: 
		X_datasets,Y_datasets = du.filtered_on_missing_indices(Y_cat)
		X_datasets = du.get_vectorized_X(X_datasets)
		nb = naive_bayes.MultinomialNB
		run_for_Y_cat(X_datasets,Y_datasets,Y_cat,nb,ALPHA_RANGE)
	


if __name__ == "__main__":
	args = sys.argv
	
	limit = None
	if len(args)>1 and str.isdigit(args[1]): 
		limit = int(args[1])
		print("Limit of ",limit)
	main(limit)


