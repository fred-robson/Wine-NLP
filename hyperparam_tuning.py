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

FILE_NAME = "results/hyper_parameters/hyper_parameters_tuning ({:%Y%m%d_%H%M%S}).csv".format(datetime.now())
POSS_LR = 10**np.random.uniform(-6, -2, 4)
POSS_EPOCHS = [50]
POSS_HIDDEN_SIZE = [50,100,200]

RESULT_INDEX = 0 #{0:Accuracy,1:F1_M,2:F1_W}


def run_for_y_cat(y_cat,config,data_helper,label_helper_points,emb_helper,limit):
    #initialize helpers


    #Pull X data
    X_train_df, X_dev_df = data_helper.X_train, data_helper.X_dev
    vocab, _ = data_helper.generate_vocab_and_word_frequencies() 
    X_train_tokens = X_train_df.as_matrix()
    X_dev_tokens = X_dev_df.as_matrix()

    #Gives appropriate indices for embeddings lookup 
    sub_emb_matrix, sub_tok2ind,sub_ind2tok, sub_unk_ind = emb_helper.get_sub_embeddings(vocab)
    X_train_indices = emb_helper.tok2ind_ind2tok(X_train_tokens, lookup_dict = sub_tok2ind, unk_indice = sub_unk_ind)
    X_dev_indices = emb_helper.tok2ind_ind2tok(X_dev_tokens, lookup_dict = sub_tok2ind, unk_indice = sub_unk_ind)
    embeddings = sub_emb_matrix
    embeddings = np.asarray(embeddings)

    #Final data_set_up 
    train_raw = [X_train_indices, label_helper_points.train_classes]
    dev_raw = [X_dev_indices, label_helper_points.dev_classes]


    #Configures the model 
    config.embed_size = embeddings.shape[1]


    #Runs the model 
    with tf.Graph().as_default():
        print("Building model...",)
        start = time.time()
        model = RNNModel(data_helper, config, embeddings,y_cat,emb_helper.test_batch,limit)
        print("took %.2f seconds"%(time.time() - start))
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            best_result_dev,corresponding_train,num_epochs = model.fit(session, saver, train_raw, dev_raw)
            print(best_result_dev)

    return best_result_dev,corresponding_train,num_epochs

def find_best_hyperparamaters(y_cat,data_helper,emb_helper,label_helper_points,limit):
    '''
    Loops through all of the parameters at the top of the file and stores the best output in 
    FILE_NAME at the top of the file 
    '''

    def set_config(config,lr,epochs,hidden_size):
        config.lr = lr
        config.n_epochs = epochs
        config.hidden_size = hidden_size
        return config

    def write_param_results(p,dev_result,train_result,num_epochs):
        with open(FILE_NAME,"a") as f: 
            for param in p: f.write(str(param)+",")
            for r in train_result: f.write(str(r)+",")
            for r in dev_result: f.write(str(r)+",")
            f.write(str(num_epochs)+"\n")

    possibilities = []
    for lr in POSS_LR:
        for epochs in POSS_EPOCHS:
            for hidden_size in POSS_HIDDEN_SIZE:
                possibilities.append((lr,epochs,hidden_size))



    param2results = {}

    for p in possibilities: 
        config = Config("lstm", n_classes = label_helper_points.num_classes, many2one = True,result_index = RESULT_INDEX)
        config = set_config(config,*p)
        best_result_dev,corresponding_train,best_epoch = run_for_y_cat(y_cat,config,data_helper,label_helper_points,emb_helper,limit)
        param2results[p] = best_result_dev
        write_param_results(p,best_result_dev,corresponding_train,best_epoch)
    
    best_params = max(param2results,key = lambda k:param2results[k][RESULT_INDEX])
    return best_params,param2results[best_params]
    



def main(limit):
    emb_helper = emb.embedding_helper(save_to_pickle = False, test_batch = 1000)
    data_helper = du.DataHelper(limit)

    f = open(FILE_NAME,"w+")
    f.write("LR,#Epochs,HS,dev_ACC,dev_F1_W,dev_F1_M,train_ACC,train_F1_W,train_F1_M,best_epoch\n")
    f.close()

    for y_cat in ["country","price","province","variety","points"]:

        sub_data_helper, sub_labels_helper = data_helper.get_filtered_data(y_cat)
        print(sub_data_helper.max_length)
        with open(FILE_NAME,"a") as f: f.write("\n"+y_cat+"\n")
        print(y_cat)
        print("*****************************")
        find_best_hyperparamaters(y_cat,sub_data_helper,emb_helper,sub_labels_helper,limit)
        print("*****************************")

if __name__ == "__main__":
    limit = None
    args = sys.argv
    if len(args)>1 and str.isdigit(args[1]): 
        limit = int(args[1])
        print("Limit of ",limit)
    main(limit)

    
