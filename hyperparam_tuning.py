from lstm import RNNModel, Config, pad_sequences
from lstm_full import MultiAttributeRNNModel
from attr_seq import Attribute2SequenceModel
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
import argparse
import os

OUTPUT_PATH = "results/hyper_parameters/"
FILE_NAME = "hyper_parameters_tuning ({:%Y%m%d_%H%M%S}).csv".format(datetime.now())
MODEL_NAME = "lstm/"
POSS_LR = [0.01]
POSS_EPOCHS = [80]
POSS_HIDDEN_SIZE = [300]

RESULT_INDEX = 0 #{0:Accuracy,1:F1_M,2:F1_W}



def run_model(config,data_helper,label_helper,emb_helper,limit, y_cat=None):
    #initialize helpers

    #Pull X data
    X_train_df, X_dev_df,X_test_df = data_helper.X_train, data_helper.X_dev, data_helper.X_test
    vocab, _ = data_helper.generate_vocab_and_word_frequencies() 
    X_train_tokens = X_train_df.as_matrix()
    X_dev_tokens = X_dev_df.as_matrix()
    X_test_tokens = X_test_df.as_matrix()

    #Gives appropriate indices for embeddings lookup 
    sub_emb_matrix, sub_tok2ind,sub_ind2tok, sub_unk_ind = emb_helper.get_sub_embeddings(data_helper.vocab)
    X_train_indices = emb_helper.tok2ind_ind2tok(X_train_tokens, lookup_dict = sub_tok2ind, unk_indice = sub_unk_ind)
    X_dev_indices = emb_helper.tok2ind_ind2tok(X_dev_tokens, lookup_dict = sub_tok2ind, unk_indice = sub_unk_ind)
    X_test_indices = emb_helper.tok2ind_ind2tok(X_test_tokens, lookup_dict = sub_tok2ind, unk_indice = sub_unk_ind)
    
    if label_helper.version == "LM":
        label_helper.update_classes_from_embeddings(sub_tok2ind, sub_unk_ind)
        config.n_classes = len(sub_emb_matrix)

    embeddings = sub_emb_matrix
    embeddings = np.asarray(embeddings)
    #Final data_set_up 
    train_raw = [X_train_indices, label_helper.train_classes]
    dev_raw = [X_dev_indices, label_helper.dev_classes]
    test_raw = [X_test_indices, label_helper.test_classes]

    #Configures the model 
    config.embed_size = embeddings.shape[1]

    #Runs the model 
    with tf.Graph().as_default():
        print("Building model...",)
        start = time.time()
        if label_helper.version == "single":
            model = RNNModel(data_helper, config, embeddings,y_cat,emb_helper.test_batch,limit,many2one=True)
        elif label_helper.version == "multi": 
            attribute_mask = label_helper.attribute_mask
            model = MultiAttributeRNNModel(data_helper, config, embeddings, attribute_mask, label_helper.attributes, emb_helper.test_batch, limit) 
        elif label_helper.version == "LM":
            model = Attribute2SequenceModel(data_helper, config, embeddings,"Language_Model",emb_helper.test_batch,limit,many2one=False)
        print("took %.2f seconds"%(time.time() - start))
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            best_result_dev,corresponding_train,num_epochs = model.fit(session, saver, train_raw, dev_raw,test_raw)
            print(best_result_dev)
            if label_helper.version == "LM":
                output_train = model.output(session, train_raw)
                output_dev = model.output(session, dev_raw)
                output_test = model.output(session, test_raw)

    return best_result_dev,corresponding_train,num_epochs

def find_best_hyperparamaters(data_helper,emb_helper,label_helper,limit, y_cat=None):
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
        with open(OUTPUT_PATH+MODEL_NAME+FILE_NAME,"a") as f: 
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
        if label_helper.version == 'single':
            config = Config("SingleAttribute","lstm", n_classes=label_helper.num_classes,result_index = RESULT_INDEX)
        elif label_helper.version == 'multi':
            config = Config("MultiAttribute","lstm", n_classes=label_helper.num_classes_max,result_index=RESULT_INDEX)
        elif label_helper.version == 'LM':
            config = Config("LanguageModel", "lstm", n_classes=len(data_helper.vocab), result_index=RESULT_INDEX)
        else:
            print("Invalid Label Helper given")
            exit()
        config = set_config(config, *p)
        best_result_dev,corresponding_train,best_epoch = run_model(config,data_helper,label_helper,emb_helper,limit, y_cat = y_cat)
        param2results[p] = best_result_dev
        write_param_results(p,best_result_dev,corresponding_train,best_epoch)
    
    best_params = max(param2results,key = lambda k:param2results[k][RESULT_INDEX])
    return best_params,param2results[best_params]
    



def main(limit, model, num_embed):
    if model == "language_model": language_model = True
    else: language_model = False

    emb_helper = emb.embedding_helper(save_to_pickle = False, test_batch=num_embed, language_model=language_model )
    data_helper = du.DataHelper(limit, language_model=language_model)

    f = open(OUTPUT_PATH+MODEL_NAME+FILE_NAME,"w+")
    f.write("LR,#Epochs,HS,dev_ACC,dev_F1_W,dev_F1_M,train_ACC,train_F1_W,train_F1_M,best_epoch\n")
    f.close()
    
    if model=="language_model":
        labels_helper = du.LabelsHelperLM(data_helper.get_data_dict(), emb_helper)
        print("*****************************")
        find_best_hyperparamaters(data_helper,emb_helper,labels_helper,limit)
        print("*****************************")
    else:
        categories =["country","price","province","variety","points"]
        if model == 'single':
            for y_cat in categories:

                sub_data_helper, sub_labels_helper = data_helper.get_filtered_data(y_cat)
                print(sub_labels_helper.train_labels)
                print(sub_labels_helper.train_classes)
                print(sub_data_helper.max_length)
                with open(OUTPUT_PATH+MODEL_NAME+FILE_NAME,"a") as f: f.write("\n"+y_cat+"\n")
                print(y_cat)
                print("*****************************")
                find_best_hyperparamaters(sub_data_helper,emb_helper,sub_labels_helper,limit, y_cat)
                print("*****************************")
        elif model == 'multi':
            labels_helper_multi = du.LabelsHelperMulti(data_helper.data_dict, categories)
            print("*****************************")
            find_best_hyperparamaters(data_helper,emb_helper,labels_helper_multi,limit)
            print("*****************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="tells us which model to run (i.e. 'single', 'multi')")
    parser.add_argument("--limit",help="sets a limit for the amount of data we want to load" ,type=int, default=None)
    parser.add_argument("--embed", help="sets a limit for number of embeddings to load", type=int, default=0)
    args = parser.parse_args()
    limit = args.limit
    model = args.model
    MODEL_NAME = model+"/"
    if not os.path.exists(OUTPUT_PATH+MODEL_NAME):
        os.makedirs(OUTPUT_PATH+MODEL_NAME)

    #args = sys.argv
    #if len(args)>1 and str.isdigit(args[1]): 
    #    limit = int(args[1])
    #    print("Limit of ",limit)
    main(limit, model, args.embed)

    
