import pandas as pd
import csv
import nltk
from Utils import assignSWNTags, check_for_idioms, adjust_idioms_tags, convert_pos_to_float
import numpy as np
from LexiconClassify import LexiconClassify
from Characters import Characters
import time

UNKNOWN_TOKEN = '<UNK>'
EMBEDDING_SIZE = 300
GLOVE_PATH = "C:\\Users\\eloib\\Downloads\\glove.twitter.27B\\glove.twitter.27B.100d.txt"
FASTTEXT_PATH = "C:\\Users\\eloib\\Downloads\\crawl-300d-2M.vec\\crawl-300d-2M.vec"
TRAIN_PATH = "C:\\Users\\eloib\\Downloads\\train\\train.csv"
CHAR_LIST = 'abcdefghijklmnopqrstuvwxyz1234567890.,\';:^()#<>[]{}!"/$%?&*'


class Preprocess:

    def __init__(self):
        t = time.time()
        print('Loading data...')
        self.train = pd.read_csv(TRAIN_PATH)
        self.x_train = self.train["comment_text"].values
        self.y_train = self.train["toxic"].values
        ### Temporary to test faster
        #self.x_train = self.x_train[0:300]
        #self.y_train = self.y_train[0:300]

        self.fasttext_embeds = pd.read_table(FASTTEXT_PATH, sep=" ", index_col=0, header=None, usecols=range(0, 301),
                                             skiprows=1, quoting=csv.QUOTE_NONE)
        self.dictionary = {}
        self.embedding_matrix = None
        self.processed_text = []
        self.targets = []
        self.lex = None
        self.char_model = Characters()
        self.vectors = []
        print('Done in '+str(time.time()-t) + 's')

    def raw_text_processing(self):
        ## take as many samples (undersampling) from each classes
        index = []
        for i in range(0, len(self.x_train)):
            if self.y_train[i] == 1:
                index.append(i)
        number_of_toxic_comments = len(index)
        i = 0
        while i < number_of_toxic_comments:
            if self.y_train[i] == 0:
                index.append(i)
            else:
                number_of_toxic_comments += 1
            i += 1

        for i in range(0, len(index)):
            sentence = self.x_train[index[i]]
            self.targets.append(self.y_train[index[i]])
            sentence = check_for_idioms(sentence)
            tokens = nltk.wordpunct_tokenize(sentence)
            tokens = [x.lower() for x in tokens]
            tokens = nltk.pos_tag(tokens)
            tokens = assignSWNTags(tokens)
            tokens = adjust_idioms_tags(tokens)
            self.processed_text.append(tokens)

    def build_embeddings(self):
        word_list = nltk.Text([item.split('#')[0] for sublist in self.processed_text for item in sublist]).vocab()
        self.embedding_matrix = np.zeros((len(word_list), EMBEDDING_SIZE))
        i = 0
        for word in word_list.keys():
            try:
                self.embedding_matrix[i] = self.fasttext_embeds.loc[word]
                self.dictionary[word.split('#')[0]] = i
                i += 1
            except KeyError:
                continue
        self.embedding_matrix[i] = np.zeros(EMBEDDING_SIZE)
        self.dictionary[UNKNOWN_TOKEN] = i
        self.embedding_matrix = self.embedding_matrix[0:i + 1]

    def lexicon_classify(self):
        lex = LexiconClassify(self.processed_text)
        lex.classify()
        self.lex = lex

    def build_vectors(self):
        t = time.time()
        print("Processing raw text...")
        self.raw_text_processing()
        print('Done in ' + str(time.time() - t) + 's')
        t = time.time()
        print("Building embeddings...")
        self.build_embeddings()
        print('Done in ' + str(time.time() - t) + 's')
        t = time.time()
        print("Looking for sentiment information...")
        self.lexicon_classify()
        print('Done in ' + str(time.time() - t) + 's')
        t = time.time()
        print("Building vectors...")
        # Each word has 300 (emb) + 60 (char) + 12 (sentiment) + 1 (pos) = 373 dimensions par mot
        self.char_model.add_char(CHAR_LIST)
        for i, sentence in enumerate(self.processed_text):
            # Without sentiment
            #sentence_vector = np.zeros((len(sentence), 360))
            # With sentiment
            sentence_vector = np.zeros((len(sentence), 373))
            for j, word in enumerate(sentence):
                pos = word[-1:]
                word = word[:-2]
                try:
                    embedding = self.embedding_matrix[self.dictionary[word]]
                except KeyError:
                    embedding = self.embedding_matrix[self.dictionary[UNKNOWN_TOKEN]]
                char_val = self.char_model.make_one_hot(word)
                sentiment = np.array(self.lex.sentiment_scores[i][j])
                # Withouth sentiment
                #word_vector = np.concatenate((embedding, char_val))
                # With sentiment
                word_vector = np.concatenate((embedding, char_val, sentiment, np.array([convert_pos_to_float(pos)])))
                sentence_vector[j] = word_vector
            self.vectors.append(sentence_vector)
        print('Done in ' + str(time.time() - t) + 's')

