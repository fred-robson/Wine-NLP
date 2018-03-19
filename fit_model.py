from hyperparam_tuning import run_model
from lstm import RNNModel, Config, pad_sequences
import embeddings as emb
import tensorflow as tf
import data_utils as du
import pandas as pd
import numpy as np
import time,sys
from datetime import datetime
from util import write_conll, print_sentence
import sklearn.metrics
import inspect

def set_config(config,lr,epochs,hidden_size):
    config.lr = lr
    config.n_epochs = epochs
    config.hidden_size = hidden_size
    return config


def fit_model(y_cat,hs,lr,epochs,limit,test_batch):
    data_helper = du.DataHelper(limit)
    sub_data_helper, sub_labels_helper = data_helper.get_filtered_data(y_cat)
    emb_helper = emb.embedding_helper(save_to_pickle = False, test_batch = test_batch)
    config = Config(RNNModel,"lstm_2_layer", n_classes = sub_labels_helper.num_classes,result_index = 0)
    config = set_config(config,lr,epochs,hs)
    run_model(config,sub_data_helper,sub_labels_helper,emb_helper,limit,y_cat)


if __name__ == "__main__":
    #Expects Y_cat hidden_size lr #epochs limit test_batch
    args = sys.argv
    y_cat = args[1]
    hs = int(args[2])
    lr = float(args[3])
    epochs = int(args[4])
    
    #Janky ways to get no arguments
    try: limit = int(args[5])
    except: limit = None

    try: test_batch = int(args[6])
    except: test_batch = False

    print("Y_cat",y_cat)
    print("Hidden Size",hs)
    print("Learning Rate",lr)
    print("epochs",epochs)
    print("lmit ",limit)
    print("test_batch",test_batch)

    fit_model(y_cat,hs,lr,epochs,limit,test_batch)





    


