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




def get_data_for_model(data_helper,label_helper,emb_helper,config):
    X_test_df = data_helper.X_test
    vocab, _ = data_helper.generate_vocab_and_word_frequencies() 
    X_test_tokens = X_test_df.as_matrix()
 
    #Gives appropriate indices for embeddings lookup 
    sub_emb_matrix, sub_tok2ind,sub_ind2tok, sub_unk_ind = emb_helper.get_sub_embeddings(vocab)
    X_train_indices = emb_helper.tok2ind_ind2tok(X_test_tokens, lookup_dict = sub_tok2ind, unk_indice = sub_unk_ind)
    embeddings = sub_emb_matrix
    embeddings = np.asarray(embeddings)

    #Final data_set_up 
    test_raw = [X_train_indices, label_helper.test_classes]
    config.embed_size = embeddings.shape[1]
    return test_raw,embeddings


def main(model_path):
    with open(model_path+"/desc.pkl","rb") as f:
        print("Loading pkl file")
        desc = pickle.load(f)
    
    y_cat = desc["Y_cat"]
    test_batch_size = desc['test_batch']
    output_path = desc['config']["output_path"]
    limit = desc["limit"]

    data_helper = du.DataHelper(limit)
    emb_helper = emb.embedding_helper(save_to_pickle = False, test_batch = test_batch_size)
    sub_data_helper, sub_labels_helper = data_helper.get_filtered_data(y_cat)

    config = Config(RNNModel,"lstm_2_layer", n_classes = sub_labels_helper.num_classes,result_index = RESULT_INDEX,output_path=output_path)

    for attribute, value in desc['config'].items(): setattr(config,attribute,value)
    
    print("Build model...")
    start = time.time()
    test_raw, embeddings = get_data_for_model(sub_data_helper,sub_labels_helper,emb_helper,config)
    model = RNNModel(data_helper,config,embeddings,many2one=True)
    print("took %.2f seconds"%(time.time() - start))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(init)
        saver.restore(session,model.config.model_output)
        results = model.evaluate(session,test_raw)
        print(results)



if __name__ == "__main__":
    args = sys.argv


    if len(args) == 1: 
        model_path = "results/lstm/"+sorted(os.listdir("results/lstm"))[-2]
    else:
        model_path = args[1]
    main(model_path)
    


