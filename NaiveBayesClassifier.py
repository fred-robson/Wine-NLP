'''
Implements Naive Bayes to run as a baseline for the wine project

'''
from data_utils import data_helper
import numpy as np
from collections import Counter,defaultdict
import sklearn.metrics
from sklearn import naive_bayes 
import sys
import pickle as pkl

ALPHA_RANGE = [1,2,3,4,5,10]
MODELS_FILE_NAME = "saved_models/NB_models.pkl"

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
    '''
    Trains a NB model and then returns the results of testing on X[test_index]
    test_index = 1 assumes that testing on dev set not test set, index = 2 for test set
    '''

    model = NB(alpha)
    x_train, y_train = X[0],Y[0]
    trained_model = train_NB(x_train,y_train,du,model)
    x_test, y_test = X[test_index], Y[test_index]
    return test_model(x_test,y_test,du,trained_model),trained_model



def train_NB(train_x,train_y,du,model,batch_size=1000):
    '''
    Trains a NB model on train_x,train_y using naive bayes model NB 
    '''
    Y_classes = set()
    for y in train_y: Y_classes|=set([y])
    Y_classes = list(Y_classes)

    
    for x_batch,y_batch in get_minibatches([train_x,train_y],batch_size):
        x_batch = du.get_vectorized_X([x_batch])
        model.partial_fit(x_batch[0],y_batch,Y_classes)
    return model

def test_model(test_x,test_y,du,model,batch_size = 1000):
    '''
    tests a model 
    '''

    Y_pred = np.array([])
    for x_batch,y_batch in get_minibatches([test_x,test_y],batch_size):
        x_batch = du.get_vectorized_X([x_batch])
        Y_pred_batch = model.predict(x_batch[0])
        Y_pred = np.append(Y_pred,Y_pred_batch)

    return test_accuracy(Y_pred,test_y)


def find_best_model(du,X,Y,NB,alpha_range):
    '''
    Assumes that X and Y are a tuple of three 
    '''
    print ("Calculating Best Alpha using dev set...")
    alpha_scores = {}
    models = {}
    for alpha in alpha_range:
        results,model = run_NB(du,alpha,X,Y,NB)
        acc,f1_w,f1_m = results
        models[alpha] = model
        alpha_scores[alpha] = f1_w
        print("Alpha ",alpha,": Acc = ",'%.3f'%acc,", F1_w= ",'%.3f'%f1_w,", F1_m= ",'%.3f'%f1_m )
    best_alpha = max(alpha_scores,key=alpha_scores.get)
    print("Best Alpha = ",best_alpha)
    return models[best_alpha]



def run_best_model(test_x,test_y,du,model,batch_size = 1000):
    acc,f1_w,f1_m = test_model(test_x,test_y,du,model,batch_size = 1000)
    print ("Calculating Test accuracy")
    print ("\rTest Accuracy: ",'%.3f'%acc)
    print ("F1 Score (weighted): ",'%.3f'%f1_w)
    print ("F1 Score (Macro): ",'%.3f'%f1_m)
    print ("\n\n")
    return model



def main(limit,save,load):
    du = data_helper(limit)
    
    to_save_models = {}

    if load:
        with open(MODELS_FILE_NAME,"rb") as f:
            saved_models = pkl.load(f)

    Y_cats = ["province","variety","country","price","points"]

    nb = naive_bayes.MultinomialNB

    for Y_cat in Y_cats: 
        X_datasets,Y_datasets = du.filtered_on_missing_indices(Y_cat)
        if Y_cat == "price": Y_datasets = du.discretize(Y_datasets)
        
        print ("Category:",Y_cat)
        print ("------------------------")

        if load: 
            model = saved_models[Y_cat]
        else:   
            model = find_best_model(du,X_datasets,Y_datasets,nb,ALPHA_RANGE)

        x_test = X_datasets[2]
        y_test = Y_datasets[2]
        run_best_model(x_test,y_test,du,model)
        to_save_models[Y_cat] = model

    if save: 
        with open(MODELS_FILE_NAME,"wb+") as f: 
            pkl.dump(to_save_models,f)



    


if __name__ == "__main__":
    args = sys.argv
    limit = None
    save = False
    load = False
    if len(args)>1 and str.isdigit(args[1]): 
        limit = int(args[1])
        print("Limit of ",limit)

    if len(args)>1 and args[1] == '-s':
        print("Models will be saved")
        save = True

    elif len(args)>1 and args[1] == "-l":
        print("Using saved models")
        load = True

    main(limit,save,load)


