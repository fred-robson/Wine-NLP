from lstm import RNNModel, Config, pad_sequences
import embeddings as emb
import tensorflow as tf
import data_utils as du
import pandas as pd
import numpy as np
import time
from datetime import date
from util import write_conll, print_sentence
import sklearn.metrics

FILE_NAME = "results/hyper_parameters/hyper_parameters_tuning (%s).csv"%date.today()
POSS_LR = [0.5,0.1,0.01]
POSS_LR = 10**np.random.uniform(-6, -2, 5)
POSS_EPOCHS = [50]
POSS_HIDDEN_SIZE = [50,100,200]

RESULT_INDEX = 0 #{0:Accuracy,1:F1_M,2:F1_W}


def run_for_y_cat(y_cat,config,data_helper,label_helper_points,emb_helper):
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
	    model = RNNModel(data_helper, config, embeddings)
	    print("took %.2f seconds"%(time.time() - start))
	    
	    init = tf.global_variables_initializer()
	    saver = tf.train.Saver()

	    with tf.Session() as session:
	        session.run(init)
	        best_result = model.fit(session, saver, train_raw, dev_raw)
	        print(best_result)

	return best_result

def find_best_hyperparamaters(y_cat,data_helper,emb_helper,label_helper_points):

	def set_config(config,lr,epochs,hidden_size):
		config.lr = lr
		config.n_epochs = epochs
		config.hidden_size = hidden_size
		return config

	def write_param_results(param,result):
		with open(FILE_NAME,"a") as f: 
			for param in p: f.write(str(param)+",")
			for r in result[:-1]: f.write(str(r)+",")
			f.write(str(result[-1])+"\n")

	possibilities = []
	for lr in POSS_LR:
		for epochs in POSS_EPOCHS:
			for hidden_size in POSS_HIDDEN_SIZE:
				possibilities.append((lr,epochs,hidden_size))


	config = Config("lstm", n_classes = label_helper_points.num_classes, many2one = True,result_index = RESULT_INDEX)
	param2results = {}

	for p in possibilities: 
		config = set_config(config,*p)
		best_result = run_for_y_cat(y_cat,config,data_helper,label_helper_points,emb_helper)
		param2results[p] = best_result
		write_param_results(p,best_result)
	
	best_params = max(param2results,key = lambda k:param2results[k][RESULT_INDEX])
	open(FILE_NAME,"a").write("Best\n")
	write_param_results(best_params,param2results[best_params])
	return best_params,param2results[best_params]
	



def main(limit):
	emb_helper = emb.embedding_helper(save_to_pickle = False, test_batch = 10000)
	data_helper = du.DataHelper(limit)

	f = open(FILE_NAME,"w+")
	f.write("LR,#Epochs,HS,ACC,F1_W,F1_M\n")
	f.close()

	for y_cat in ["price","province","variety","country","points"]:

		label_helper_points = data_helper.labels_from_Y_cat(y_cat)
		with open(FILE_NAME,"a") as f: f.write("\n"+y_cat+"\n")

		print(y_cat)
		print("*****************************")
		find_best_hyperparamaters(y_cat,data_helper,emb_helper,label_helper_points)
		print("*****************************")

if __name__ == "__main__":
	limit=None
	if len(args)>1 and str.isdigit(args[1]): 
        limit = int(args[1])
        print("Limit of ",limit)
	main(limit)

	