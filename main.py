from lstm import RNNModel, Config, pad_sequences
import embeddings as emb
import tensorflow as tf
import data_utils as du
import pandas as pd
import numpy as np
import time
from util import write_conll, print_sentence

def run_for_y_cat(y_cat,num_hidden=None,limit=None):
	#initialize helpers
	emb_helper = emb.embedding_helper(save_to_pickle = False, test_batch = 10000)
	data_helper = du.DataHelper(limit)
	label_helper_points = data_helper.labels_from_Y_cat(y_cat)

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
	config = Config("lstm", n_classes = label_helper_points.num_classes, many2one = True)
	config.embed_size = embeddings.shape[1]


	#Runs the model 
	with tf.Graph().as_default():
	    print("Building model...",)
	    start = time.time()
	    model = RNNModel(data_helper, config, embeddings)
	    print("took %.2f seconds", time.time() - start)
	    
	    init = tf.global_variables_initializer()
	    saver = tf.train.Saver()

	    with tf.Session() as session:
	        session.run(init)
	        model.fit(session, saver, train_raw, dev_raw)
	        output = model.output(session, dev_raw)
	        sentences, class_labels, predictions = zip(*output)
	        predictions = [[str(label_helper_points.class_2_lbl[cls]) for cls in preds] for preds in predictions]
	        labels = [[str(label_helper_points.class_2_lbl[cls]) for cls in classes] for classes in class_labels]
	        label_results = zip(labels, predictions)
	        sentences = emb_helper.tok2ind_ind2tok(sentences, lookup_dict = sub_ind2tok, unk_indice = sub_unk_ind)
	        output = zip(sentences, labels, predictions)

if __name__ == "__main__":
	for y_cat in ["province","variety","country","points"]:
		print(y_cat)
		print("-----------------------------")
		run_for_y_cat(y_cat,limit=500)
		print("-----------------------------")