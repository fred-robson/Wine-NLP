# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
'''
File for accessing and manipulating the data
'''
from defs import NAN_TOK, START_TOK, END_TOK, LANGUAGE_MODEL_ATTRIBUTES
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


def process_frame_for_language_model(data_frame):
    def add_start_and_end_tokens(row):
        row.append(END_TOK)
        row.insert(0, START_TOK)
        return row
    full_cols = [X_cat] + LANGUAGE_MODEL_ATTRIBUTES
    data_frame = data_frame[full_cols]    
    for attribute in LANGUAGE_MODEL_ATTRIBUTES:
        data_frame = data_frame.loc[data_frame[attribute] != NAN_TOK]
    data_frame[X_cat] = data_frame[X_cat].apply(add_start_and_end_tokens)
    return data_frame

def data_frame_as_list(df):
    return [[row] for row in df.as_matrix()]

def frame_dict_to_list_dict(frame_dict, frame=None):
    '''
    args: a dictionary of data frames
    return: a dictionary of list equivalents of the data frames
    '''
    list_dict = {}
    for key, value in frame_dict.items():
        if frame:
            list_dict[key] = value[frame].as_matrix()
        else:
            list_dict[key] = value[frame].as_matrix()
    return list_dict

def descritize(data, y_cat,  k = 20):
    '''
    assumes this is data of form [train_df, dev_df, test_df]
    returns: [train_df, dev_tf, test_df] (discretized)
    '''
    frames = pd.concat(data.copy())
    frames = frames[~frames.isin([NAN_TOK])]
    model = KMeans(n_clusters = k)
    model.fit(frames.as_matrix().reshape(-1,1))
    for i, frame in enumerate(data):
        null_mask = frame.ne(NAN_TOK)
        mask = null_mask.astype('int64')
        frame_fill = frame.replace(NAN_TOK, -1)
        cat_array = model.predict(frame_fill.copy().as_matrix().reshape(-1,1))
        cat_array = cat_array.astype(int)
        cat_array = cat_array.astype(object)
        data_i = frame.copy(deep=True)
        df = pd.DataFrame(data_i).astype(object)
        df[y_cat] = cat_array
        df[frame.isin([NAN_TOK])] = NAN_TOK
        data[i] = df[y_cat]
    return data

class LabelsHelperLM():
    version = 'LM'
    def __init__(self, batch_dict, emb_helper):
        self.batch_dict = batch_dict
        self.labels_frame_batch_dict = self.input_dict_to_label_dict()
        self.emb_helper = emb_helper
        self.labels_list_dict = frame_dict_to_list_dict(self.labels_frame_batch_dict, frame=X_cat) 
        self.classes_list_dict = None
        self.train_classes = None
        self.dev_classes = None
        self.test_classes = None 

    def update_classes_from_embeddings(self, tok2ind, unk_ind):
        classes_list_dict = {}
        for key, value in self.labels_list_dict.items():
            classes_list_dict[key] = self.emb_helper.tok2ind_ind2tok(value, lookup_dict=tok2ind, unk_indice=unk_ind)
        self.classes_list_dict = classes_list_dict
        self.train_classes = self.classes_list_dict["train"]
        self.dev_classes = self.classes_list_dict["dev"]
        self.test_classes = self.classes_list_dict["test"]

    def input_dict_to_label_dict(self):
        labels_dict = {}
        for key, value in self.batch_dict.items():
            labels_dict[key] = self.input_data_frame_to_label_data_frame(value) 
        return labels_dict

    @staticmethod
    def input_data_frame_to_label_data_frame(data_frame):
        def add_end_token_remove_start_token(row):
            row_copy = copy.deepcopy(row)
            row_copy.append(END_TOK)
            row_copy.pop(0)
            return row_copy
        data_frame_copy = data_frame.copy(deep=True)
        data_frame_copy[X_cat] = data_frame_copy[X_cat].copy(deep=True).apply(add_end_token_remove_start_token)
        return data_frame_copy

class LabelsHelperMulti():
    num_classes_max = 0 # the number of classes of the attribute with the most classes (gets set in init)
    version = 'multi'
    def __init__(self, batch_dict, attributes):
        '''
        batch_dict is dict of form: {"train" : data_frame, "dev" : data_frame, "test" : data_frame}
        where data_frame is the full data_frame
        '''
        self.attributes = attributes
        self.batch_dict = batch_dict
        self.helper_dict = self.generate_label_helper_dict()
        self.labels_list_dict = self.group_attributes_as_list()
        self.classes_list_dict = self.group_attributes_as_list(as_classes = True)
        self.attribute_mask = self.create_attribute_mask()
        self.train_classes = self.classes_list_dict["train"]
        self.dev_classes = self.classes_list_dict["dev"]
        self.test_classes = self.classes_list_dict["test"]

    def generate_label_helper_dict(self):    
        lbl_helper_dict = {}
        x_cat = "description"
        for category in self.attributes:
            sub_batch_dict = copy.deepcopy(self.batch_dict)
            sub_batch_dict["train"] = sub_batch_dict["train"][category]
            sub_batch_dict["dev"] = sub_batch_dict["dev"][category]
            sub_batch_dict["test"] = sub_batch_dict["test"][category]
            lbl_helper = LabelsHelper(sub_batch_dict, category)
            lbl_helper_dict[category] = lbl_helper
            if lbl_helper.num_classes > self.num_classes_max:
                self.num_classes_max = lbl_helper.num_classes
        return lbl_helper_dict

    def group_attributes_as_list(self, as_classes = False):
        attribute_dict = {}
        for cat in self.attributes: 
            helper = self.helper_dict[cat]
            if not as_classes:
                list_dict = helper.labels_list_dict 
            else:
                list_dict = helper.classes_list_dict
            for key, value in list_dict.items():
                if attribute_dict.get(key, "") is "":
                    attribute_dict[key] = [value]
                else:
                    attribute_dict[key].append(value)
        for key, value in attribute_dict.items():
            attribute_dict[key] = list(map(list, zip(*value)))
        return attribute_dict

    def create_attribute_mask(self):
        mask = np.zeros((len(self.attributes),self.num_classes_max ))
        for i,attribute in enumerate(self.attributes):
            num_classes = self.helper_dict[attribute].num_classes
            mask[i,1:num_classes] = 1
        return mask

class LabelsHelper():
    version = 'single'
    def __init__(self, batch_dict, Y_cat):
        '''
        batch_dict is dict of form: {"train" : data_frame, "dev" : data_frame, "test" : data_frame}
        '''
        self.name = Y_cat
        if Y_cat == "price":
            data = [batch_dict["train"], batch_dict["dev"], batch_dict["test"]]
            data = descritize(data, Y_cat, k=10)
            batch_dict["train"], batch_dict["dev"], batch_dict["test"] = data[0], data[1], data[2]
        self.batch_dict = batch_dict
        self.train_df = batch_dict["train"]
        self.dev_df = batch_dict["dev"]
        self.test_df = batch_dict["test"]
        self.train_labels = data_frame_as_list(self.train_df)
        self.dev_labels =data_frame_as_list(self.dev_df)
        self.test_labels = data_frame_as_list(self.test_df)
        self.labels_list_dict = {"train" : self.train_labels, "dev" : self.dev_labels, "test": self.test_labels}
        self.lbl_2_class, self.class_2_lbl, self.num_classes = self.characterize_labels()
        self.train_classes = self.label_list_2_class_list(self.train_labels)
        self.dev_classes =self.label_list_2_class_list(self.dev_labels)
        self.test_classes =self.label_list_2_class_list(self.test_labels)
        self.classes_list_dict = {"train": self.train_classes, "dev" : self.dev_classes, "test" : self.test_classes}

    
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
        lbl_2_class[NAN_TOK] = _class # the 0-th class is reserved for missing labels
        class_2_lbl[_class] = NAN_TOK
        for batch in [self.train_labels, self.dev_labels, self.test_labels]:
            for label in batch:
                if type(label) is list:
                    label = label[0]
                if lbl_2_class.get(label, "") is "":
                    _class+=1
                    lbl_2_class[label] = _class
                    class_2_lbl[_class] = label
        return lbl_2_class, class_2_lbl, _class+1

class DataHelper():

    def __init__(self,max_len=None, data = None, language_model = False):
        '''
        Max-len is useful for testing 
        '''
        self.max_length = 0
        self.limit = max_len
        if data is None:
            self.train_data,self.dev_data,self.test_data = self.load_data(max_len=self.limit, language_model=language_model)
        else: 
            self.train_data,self.dev_data,self.test_data = data
        if language_model:
            self.train_attr_df, self.dev_attr_df, self.test_attr_df = None,None,None
            self.attributes_t2i,self.attributes_i2t, self.n_attribute_classes = self.characterize_attributes()
            self.attributes_train, self.attributes_dev, self.attributes_test = self.encode_attributes()
        self.X_train = self.train_data[X_cat]
        self.X_dev = self.dev_data[X_cat]
        self.X_test = self.test_data[X_cat]
        self.data_dict = {"train": self.train_data, "dev" : self.dev_data, "test" : self.test_data}
        self.vocab, self.word_freq_dict = self.generate_vocab_and_word_frequencies()
        self.vocab_to_index = {v:i for i,v in enumerate(sorted(list(self.vocab)))}
        

    def characterize_attributes(self):

        attributes = [self.train_data[LANGUAGE_MODEL_ATTRIBUTES], self.dev_data[LANGUAGE_MODEL_ATTRIBUTES], self.test_data[LANGUAGE_MODEL_ATTRIBUTES]]
        price_frames  = descritize([attributes[0]["price"], attributes[1]["price"], attributes[2]["price"]], "price")
        attributes[0]["price"], attributes[1]["price"], attributes[2]["price"] = price_frames[0], price_frames[1], price_frames[2]
        self.train_attr_df = attributes[0]
        self.dev_attr_df = attributes[1]
        self.test_attr_df = attributes[2]
        af = pd.concat(attributes)
        attr2ind = {}
        ind2attr = {}
        encoding = 0
        for key, value in af.iteritems():
            for ind, attr in value.iteritems():
                if attr2ind.get(attr, NAN_TOK) is NAN_TOK:
                    attr2ind[attr] = encoding
                    ind2attr[encoding] = attr
                    encoding += 1
        return attr2ind,ind2attr, encoding

    def encode_attributes(self):
        train_attr = self.train_attr_df.as_matrix()
        dev_attr = self.dev_attr_df.as_matrix()
        test_attr = self.test_attr_df.as_matrix()
        return self.get_encodings(train_attr), self.get_encodings(dev_attr), self.get_encodings(test_attr)

    def get_encodings(self, mat):
        mat_encoded = np.zeros_like(mat, dtype = np.int32)
        for i,row in enumerate(mat):
            for j, attr in enumerate(row):
                mat_encoded[i, j] = self.attributes_t2i[attr]
        return mat_encoded

    def load_data(self,max_len=None, language_model=False):
        '''
        Loads the data from the pickle file. Called at initialization 
        returns: train_data,dev_data,test_data
        '''
        loaded_data = []
        for filename in [train_data_file,dev_data_file,test_data_file]:
            data_frame = pd.read_pickle(filename)
            data_frame = data_frame.fillna(NAN_TOK)
            if language_model:
                data_frame = process_frame_for_language_model(data_frame)
            if max_len == None: loaded_data.append(data_frame)
            else: loaded_data.append(data_frame[:max_len])
        return loaded_data

    def get_data_dict(self):
        train_copy = self.train_data.copy(deep=True)
        dev_copy = self.dev_data.copy(deep=True)
        test_copy = self.test_data.copy(deep=True)
        return {"train": train_copy, "dev":dev_copy, "test":test_copy}
    def get_all_data(self,category):
        #returns the concatenated 
        all_data = np.array([])
        for data in [self.train_data,self.dev_data,self.test_data]:
            all_data = np.append(all_data,data[category])
        return all_data

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
        returns: new DataHelper and new LabelsHelper without examples that have corresponding labels which are NaN
        '''
        X_cat = "description"
        new_cats = [X_cat, Y_cat]
        train_df, dev_df, test_df = self.train_data[new_cats].copy(),   self.dev_data[new_cats].copy(),  self.test_data[new_cats].copy()
        train_df, dev_df, test_df = train_df[~train_df[Y_cat].isin([NAN_TOK])], dev_df[~dev_df[Y_cat].isin([NAN_TOK])], test_df[~test_df[Y_cat].isin([NAN_TOK])]
        data_frames = [train_df, dev_df, test_df]
        if type(train_df[Y_cat].dtype) is np.float64:
            data_frames = descritize(data_frames,Y_cat)
        return DataHelper(data = data_frames), self.labels_from_Y_cat(Y_cat, data_frames = data_frames, filter_nan = True)
   

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
        assert len(data) == 2 or len(data) ==3, ("data must be of form [examples, labels]")
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
                if type(y_i) is str and y_i == NAN_TOK: 
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

if __name__ == "__main__": 
    du =DataHelper(language_model = True)
    train = du.train_data
    print(train)
    #print(train["region_2"].as_matrix())
    #features = ["region_2", "price"]
    #print(train["price"].dtype)
    #batch_dict = du.data_dict
    #print(batch_dict["train"])
    #lh_multi = LabelsHelperMulti(batch_dict, features)
    #sub_du, label_help = du.get_filtered_data("price")
    #train_df = sub_du.train_data
    #print(train_df)
    #print(sub_du.descritize([train_df, train_df, train_df], y_cat = "price"))
    #print("") 

    #d = {'col1': ["hi", "hi", "hi"], 'col2': [np.NaN, "sup", "woah"], 'col3': [81, 74, 90]}
    #df = pd.DataFrame(data=d)
    #print(df)
    #g = df.as_matrix()
    #print(df.as_matrix())
    #print(type(g[0, 0]))
    #print(type(g[0, 1]))
    #batch_dict = {"train" : df, "dev" : df, "test" : df}
    #categories = ['col1', 'col2', 'col3']
    #lh_multi = LabelsHelperMulti(batch_dict, categories)
    #print(lh_multi.num_classes_max)
    #print(lh_multi.labels_list_dict['train'])
    #print(lh_multi.classes_list_dict['train'])
    #print(lh_multi.attribute_mask.shape)
    #freq_dict = du.word_freq_dict
    #print (json.dumps(freq_dict, indent=1))
    #print(len(du.vocab))
    #points = du.labels_from_Y_cat("points")
    #print(points.train_classes)
    #print(points.num_classes)
    #print(du.discretize("price",20))
