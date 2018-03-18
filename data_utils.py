# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
'''
File for accessing and manipulating the data
'''

import pandas as pd
import pprint
import numpy as np
from collections import Counter,defaultdict
import copy,math
import json
from sklearn.cluster import KMeans

train_data_file = "data/train_utf.pkl"  
dev_data_file = "data/dev_utf.pkl"
test_data_file = "data/test_utf.pkl"

X_cat = "description"


def data_frame_as_list(df):
    return [[row] for row in df.as_matrix()]


class LabelsHelper():

    def __init__(self, batch_dict, Y_cat):
        '''
        batch_dict is dict of form: {"train" : data_frame, "dev" : data_frame, "test" : data_frame}
        '''
        self.name = Y_cat
        self.train_df = batch_dict["train"]
        self.dev_df = batch_dict["dev"]
        self.test_df = batch_dict["test"]
        self.train_labels = data_frame_as_list(self.train_df)
        self.dev_labels =data_frame_as_list(self.dev_df)
        self.test_labels = data_frame_as_list(self.test_df)
        self.lbl_2_class, self.class_2_lbl, self.num_classes = self.characterize_labels()
        self.train_classes = self.label_list_2_class_list(self.train_labels)
        self.dev_classes =self.label_list_2_class_list(self.dev_labels)
        self.test_classes =self.label_list_2_class_list(self.test_labels)

    def label_list_2_class_list(self, label_list):
        class_list = []
        for label in label_list:
            if type(label) is list:
                label = label[0]
            _class = self.lbl_2_class[label]
            class_list.append([_class])
        return class_list

    def characterize_labels(self):
        lbl_2_class = {}
        class_2_lbl = {}
        _class = 0
        for batch in [self.train_labels, self.dev_labels, self.test_labels]:
            for label in batch:
                if type(label) is list:
                    label = label[0]
                if lbl_2_class.get(label, "") is "":
                    lbl_2_class[label] = _class
                    class_2_lbl[_class] = label
                    _class += 1
        return lbl_2_class, class_2_lbl, _class


class DataHelper():

    def __init__(self,max_len=None, data = None):
        '''
        Max-len is useful for testing 
        '''
        self.max_length = 0
        self.limit = max_len
        if data is None:
            self.train_data,self.dev_data,self.test_data = self.load_data(max_len=self.limit)
        else: 
            self.train_data,self.dev_data,self.test_data = data
        self.X_train = self.train_data[X_cat]
        self.X_dev = self.dev_data[X_cat]
        self.X_test = self.test_data[X_cat]
        self.vocab, self.word_freq_dict = self.generate_vocab_and_word_frequencies()
        self.vocab_to_index = {v:i for i,v in enumerate(sorted(list(self.vocab)))}
        
    def load_data(self,max_len=None):
        '''
        Loads the data from the pickle file. Called at initialization 
        returns: train_data,dev_data,test_data
        '''
        loaded_data = []
        for filename in [train_data_file,dev_data_file,test_data_file]:
            data_frame = pd.read_pickle(filename)
            if max_len == None: loaded_data.append(data_frame)
            else: loaded_data.append(data_frame[:max_len])
        return loaded_data

    def get_Y_cat(self,Y_cat, data_frames = None):
        if data_frames is None:
            return self.train_data[Y_cat],self.dev_data[Y_cat],self.test_data[Y_cat]
        else:
            return data_frames[0][Y_cat],data_frames[1][Y_cat],data_frames[2][Y_cat]
    def labels_from_Y_cat(self, Y_cat, data_frames = None, filter_nan = False):
        '''
        returns: train_labels, dev_labels, test_labels, dict = {label : class}, num_classes  
        '''
        train_df, dev_df, test_df = self.get_Y_cat(Y_cat, data_frames = data_frames)
        batch_dict = {"train" : train_df, "dev": dev_df, "test": test_df }
        return LabelsHelper(batch_dict, Y_cat)

    def get_filtered_data(self, Y_cat):
        '''
        returns: new cleaned data frames containing for train, dev, test and corresponding labels
        '''
        X_cat = "description"
        new_cats = [X_cat, Y_cat]
        train_df, dev_df, test_df = self.train_data[new_cats].copy().dropna(axis=0),   self.dev_data[new_cats].copy().dropna(axis=0),  self.test_data[new_cats].copy().dropna(axis=0)
        data_frames = [train_df, dev_df, test_df]
        return DataHelper(data = data_frames), self.labels_from_Y_cat(Y_cat, data_frames = data_frames, filter_nan = True)
   
   
    def descritize(self, data, y_cat,  k = 20):
        '''
        assumes this is data of form [train_df, dev_df, test_df]
        '''
        frames = pd.concat(data.copy())
        model = KMeans(n_clusters = k)
        model.fit(frames[y_cat].as_matrix().reshape(-1,1))
        for i, frame in enumerate(data):
           data[i][y_cat] = model.predict(frame[y_cat].copy().as_matrix().reshape(-1,1))
        return data

        """
        def discretize(self,Y,num_categories=20):
        '''
        Converts a continous distribution into an equally split discrete distribution
        Assumes Y is a tuple of [train,dev,test]
        '''        
        all_data = np.array([])
        for data in Y:
            all_data = np.append(all_data,data)
        all_data = all_data.astype(float)
        all_data = np.sort(all_data)
        all_data = all_data[~np.isnan(all_data)]

        discrete_areas = np.array_split(all_data,num_categories)
        bins = [b[0] for b in discrete_areas]
        bins.append(float("inf"))

        ret = []
        for cat_data in Y:
            discretized = []
            for y in cat_data:
                if math.isnan(y): 
                    discretized.append(None)
                else:
                    discretized.append(str(min(set(x for x in bins if x>y))))
            ret.append(discretized)

        return ret
        """


    def generate_vocab_and_word_frequencies(self):
        '''
        Generates the vocabulary and word frequencies in train, test and dev
        Returns: set of words making up vocab, dictionary of word frequencies {word: count}
        '''
        vocab = set()
        word_counts = defaultdict(int)
        for data in [self.X_train,self.X_dev,self.X_test]:
            for desc in data:
                if len(desc) > self.max_length:
                    self.max_length = len(desc)
                for word in desc:
                    vocab.add(word)
                    word_counts[word]+=1 
        return vocab, word_counts
   
    def data_as_list_of_tuples(self, data):
        """
        data is of form [sentences, labels]
        return [(sentences[0], labels[0]), (sentences[1], labels[1]) ...]
        """
        assert len(data) == 2, ("data must be of form [examples, labels]")
        data_as_list = []
        for data_tup in zip(*data):       
            data_as_list.append(data_tup)
        return data_as_list

    def missing_indices(self,Y_cat):
        '''
        Returns a vector of indices where the Y category is empty
        ''' 
        ret = []
        Y_data = self.get_Y_cat(Y_cat)
        for Y in Y_data:
            indices = []
            for i,y_i in enumerate(Y):
                if not type(y_i) is str and math.isnan(y_i): 
                    indices.append(i)
            ret.append(indices)
        return ret 

    def filtered_on_missing_indices(self,Y_cat):
        #Removes anything that has a missing Y value
        X = self.X_train, self.X_dev, self.X_test
        Y = self.get_Y_cat(Y_cat)
        Z = self.missing_indices(Y_cat)

        ret = [],[]

        for x,y,z in zip(X,Y,Z):
            ret[0].append(np.delete(np.array(x),z))
            ret[1].append(np.delete(np.array(y),z))
        return ret
    

    def get_vectorized_X(self,all_X=None):
        '''
        Converts the ["I","am"...."am"] into a vector where the index represents a word count
        '''
        ret = []


        if all_X is None: 
            all_X =[self.X_train,self.X_dev,self.X_test] 

        for X  in all_X:
            X_vectorized = np.zeros([len(X),len(self.vocab)])
            for i,row in enumerate(X):
                for w in row:
                    j = self.vocab_to_index[w]
                    X_vectorized[(i,j)]+=1
            ret.append(X_vectorized)
        return ret  

    """
    def generate_vocab_and_word_frequencies(self):
        '''
        Generates the vocabulary and word frequencies in train, test and dev
        Returns: set of words making up vocab, dictionary of word frequencies {word: count}
        '''
        vocab = set()
        word_counts = defaultdict(int)
        for data in [self.X_train,self.X_dev,self.X_test]:
            for desc in data:
                if len(desc) > self.max_length:
                    self.max_length = len(desc)
                for word in desc:
                    vocab.add(word)
                    word_counts[word]+=1 
        return vocab, word_counts
        """
if __name__ == "__main__":
    du =DataHelper(100)
    train = du.train_data
    feature = ["price"]
    sub_du, label_help = du.get_filtered_data("price")
    train_df = sub_du.train_data
    print(sub_du.descritize([train_df, train_df, train_df], y_cat = "price"))
    print("")        

    #freq_dict = du.word_freq_dict
    #print (json.dumps(freq_dict, indent=1))
    #print(len(du.vocab))
    #points = du.labels_from_Y_cat("points")
    #print(points.train_classes)
    #print(points.num_classes)
    #print(du.discretize("price",20))
