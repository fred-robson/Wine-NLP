from __future__ import print_function
import os, sys
import embeddings as emb
import data_utils as du
import time
from lstm import RNNModel, Config, pad_sequences
import numpy as np
import pickle
from hyperparam_tuning import RESULT_INDEX
import tensorflow as tf

DEFAULT_FOLDER = "results/<class 'lstm.RNNModel'>/lstm_2_layer"
DATA_TYPE = "train"



def get_data_for_model(data_helper,label_helper,emb_helper,config,data_type="test"):
    if data_type=="train": 
        X_data_df = data_helper.X_dev
    if data_type=="dev":
        X_data_df = data_helper.X_dev
    else:
        X_data_df = data_helper.X_test
    
    vocab, _ = data_helper.generate_vocab_and_word_frequencies() 
    X_data_tokens = X_data_df.as_matrix()
 
    #Gives appropriate indices for embeddings lookup 
    sub_emb_matrix, sub_tok2ind,sub_ind2tok, sub_unk_ind = emb_helper.get_sub_embeddings(vocab)
    X_data_indices = emb_helper.tok2ind_ind2tok(X_data_tokens, lookup_dict = sub_tok2ind, unk_indice = sub_unk_ind)
    embeddings = sub_emb_matrix
    embeddings = np.asarray(embeddings)

    if data_type=="train": 
        klasses = label_helper.train_classes
    if data_type=="dev":
        klasses = label_helper.dev_classes
    else:
        klasses = label_helper.test_classes

    #Final data_set_up 
    data_raw = [X_data_indices, klasses]
    config.embed_size = embeddings.shape[1]
    return data_raw,embeddings


def main(model_path):

    print("Model Path:",model_path)
    with open(model_path+"/desc.pkl","rb") as f:
        print("Loading pkl file")
        desc = pickle.load(f)
    

    y_cat = desc["Y_cat"]
    test_batch_size = desc['test_batch']
    output_path = desc['config']["output_path"]
    limit = desc["limit"]

    print("Running for "+DATA_TYPE)
    print("Y-cat",y_cat)
    print("Limit",limit)
    print("Batch Size",test_batch_size)

    data_helper = du.DataHelper(limit)
    emb_helper = emb.embedding_helper(save_to_pickle = False, test_batch = test_batch_size)
    sub_data_helper, sub_labels_helper = data_helper.get_filtered_data(y_cat)

    with tf.Graph().as_default():

        config = Config(RNNModel,"lstm_2_layer", n_classes = sub_labels_helper.num_classes,result_index = RESULT_INDEX,output_path=output_path)

        for attribute, value in desc['config'].items(): 
            setattr(config,attribute,value)

    
        start = time.time()
        print("Building model....")
        test_raw, embeddings = get_data_for_model(sub_data_helper,sub_labels_helper,emb_helper,config,DATA_TYPE)
        model = RNNModel(data_helper, config, embeddings,y_cat,emb_helper.test_batch,limit,many2one=True)
        print(model.config.model_output)
        print("took %.2f seconds"%(time.time() - start))
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session,model.config.model_output)
            results = model.evaluate(session,test_raw)
            print(results)



if __name__ == "__main__":
    args = sys.argv


    if len(args) == 1: 
        model_path = DEFAULT_FOLDER+"/"+sorted(os.listdir(DEFAULT_FOLDER))[-1]
    else:
        model_path = args[1]
    main(model_path)
    


