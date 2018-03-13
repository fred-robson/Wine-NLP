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

def get_minibatches(data, minibatch_size, shuffle=False):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)

def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

def minibatches(data, batch_size, shuffle=False):
    batches = [np.array(col) for col in zip(*data)]
    return self.get_minibatches(batches, batch_size, shuffle)		


def test_accuracy(Y_pred,Y_true):
	acc = sklearn.metrics.accuracy_score(Y_pred,Y_true)
	f1_w  = sklearn.metrics.f1_score(Y_pred,Y_true,average="weighted")	
	f1_m = sklearn.metrics.f1_score(Y_pred,Y_true,average="macro")	
	return acc,f1_w,f1_m

def run_NB(du,alpha,X,Y,NB,test_index = 1,batch_size = 1000):
	#test_index = 1 assumes that testing on dev set not test set 
	
	naive_bayes = NB(alpha)
	
	Y_classes = set()
	for y in Y: Y_classes|=set(y)
	Y_classes = list(Y_classes)
	
	for x_batch,y_batch in get_minibatches([X[0],Y[0]],batch_size):
		x_batch = du.get_vectorized_X([x_batch])
		naive_bayes.partial_fit(x_batch[0],y_batch,Y_classes)

	Y_pred = np.array([])
	for x_batch,y_batch in get_minibatches([X[test_index],Y[test_index]],batch_size):
		x_batch = du.get_vectorized_X([x_batch])
		Y_pred_batch = naive_bayes.predict(x_batch[0])
		Y_pred = np.append(Y_pred,Y_pred_batch)

	Y_true = Y[test_index]	
	return test_accuracy(Y_pred,Y_true)

def run_for_Y_cat(du,X,Y,Y_cat,NB,alpha_range):
	'''
	Assumes that X and Y are a tuple of three 
	'''
	print ("Category:",Y_cat)
	print ("------------------------")
	print ("Calculating Best Alpha using dev set...")
	alpha_scores = {}
	for alpha in alpha_range:
		acc,f1_w,f1_m = run_NB(du,alpha,X,Y,NB)
		alpha_scores[alpha] = f1_w
		print("Alpha ",alpha,": Acc = ",'%.3f'%acc,", F1_w= ",'%.3f'%f1_w,", F1_m= ",'%.3f'%f1_m )
	best_alpha = max(alpha_scores,key=alpha_scores.get)
	print("Best Alpha = ",best_alpha)
	print ("Calculating Test accuracy")
	acc,f1_w,f1_m = run_NB(du,best_alpha,X,Y,NB,2)
	print ("\rTest Accuracy: ",'%.3f'%acc)
	print ("F1 Score (weighted): ",'%.3f'%f1_w)
	print ("F1 Score (Macro): ",'%.3f'%f1_m)
	print ("\n\n")



def main(limit):
	du = data_helper(limit)
	
	Y_cats = ["province","variety","country","price","points"]

	for Y_cat in Y_cats: 
		X_datasets,Y_datasets = du.filtered_on_missing_indices(Y_cat)
		if Y_cat == "price": Y_datasets = du.discretize(Y_datasets)
		nb = naive_bayes.MultinomialNB
		run_for_Y_cat(du,X_datasets,Y_datasets,Y_cat,nb,ALPHA_RANGE)
	


if __name__ == "__main__":
	args = sys.argv
	
	limit = None
	if len(args)>1 and str.isdigit(args[1]): 
		limit = int(args[1])
		print("Limit of ",limit)
	main(limit)


