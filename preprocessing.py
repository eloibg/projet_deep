import pandas as pd
import csv
import nltk
from Utils import assignSWNTags, check_for_idioms, adjust_idioms_tags, convert_pos_to_float
import numpy as np
from LexiconClassify import LexiconClassify
from Characters import Characters
import time
from AddNoise import add_noise

UNKNOWN_TOKEN = '<UNK>'
EMBEDDING_SIZE = 300
CHAR_LIST = 'abcdefghijklmnopqrstuvwxyz1234567890.,\';:^()#<>[]{}!"/$%?&*'


class Preprocess:

    def __init__(self, TRAIN_PATH, FASTTEXT_PATH, sentiment=True):
        t = time.time()
        print('Loading data...')
        self.train = pd.read_csv(TRAIN_PATH)
        self.x_train = self.train["comment_text"].values
        self.y_train = self.train["toxic"].values

        self.fasttext_embeds = pd.read_table(FASTTEXT_PATH, sep=" ", index_col=0, header=None, usecols=range(0, 301),
                                             skiprows=1, quoting=csv.QUOTE_NONE)#, nrows=100)
        self.dictionary = {}
        self.embedding_matrix = None
        self.processed_text = []
        self.targets = []
        self.lex = None
        self.char_model = Characters()
        self.vectors = []
        self.sentiment = sentiment
        print('Done in '+ str(time.time()-t) + 's')

    def __undersampling__(self):
        ### Take same amounts of samples from each classes
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
        return index

    def raw_text_processing(self):
        index = self.__undersampling__()
        for i in range(0, len(index)):
            sentence = self.x_train[index[i]]
            toxic = self.y_train[index[i]]
            self.targets.append(self.y_train[index[i]])
            sentence = check_for_idioms(sentence)
            tokens = nltk.wordpunct_tokenize(sentence)
            tokens = [x.lower() for x in tokens]
            if toxic == 1:
                tokens = add_noise(tokens)
            tokens = nltk.pos_tag(tokens)
            tokens = assignSWNTags(tokens)
            tokens = adjust_idioms_tags(tokens)
            self.processed_text.append(tokens)

    def build_embeddings(self):
        word_list = nltk.Text([item for sublist in self.processed_text for item in sublist]).vocab()
        if self.sentiment:
            self.embedding_matrix = np.zeros((len(word_list), EMBEDDING_SIZE+12))
        else:
            self.embedding_matrix = np.zeros((len(word_list), EMBEDDING_SIZE))
        i = 0
        for word in word_list.keys():
            try:
                embed = self.fasttext_embeds.loc[word.split('#')[0]]
                if self.sentiment:
                    sentiment = self.lex.word_sentiment(word)
                    self.embedding_matrix[i] = np.concatenate((embed, sentiment))
                else:
                    self.embedding_matrix[i] = np.concatenate((embed))
                self.dictionary[word] = i
                i += 1
            except KeyError:
                continue
        self.embedding_matrix[i] = np.zeros(len(self.embedding_matrix[0]))
        self.dictionary[UNKNOWN_TOKEN] = i
        self.embedding_matrix = self.embedding_matrix[0:i + 1]

    def lexicon_classify(self):
        lex = LexiconClassify(self.processed_text)
        lex.open_dicts()
        self.lex = lex

    def build_vectors(self):
        t = time.time()
        print("Processing raw text...")
        self.raw_text_processing()
        print('Done in ' + str(time.time() - t) + 's')
        t = time.time()
        if self.sentiment:
            t = time.time()
            print("Looking for sentiment information...")
            self.lexicon_classify()
            print('Done in ' + str(time.time() - t) + 's')
        print("Building embeddings...")
        self.char_model.add_char(CHAR_LIST)
        self.build_embeddings()
        print('Done in ' + str(time.time() - t) + 's')
        t = time.time()
        print("Building vectors...")
        for i, sentence in enumerate(self.processed_text):
            # +2 car token UNK et index de embedding
            sentence_vector = np.zeros((len(sentence), len(CHAR_LIST)+2))
            for j, word in enumerate(sentence):
                if word in self.dictionary:
                    index = self.dictionary[word]
                else:
                    index = self.dictionary[UNKNOWN_TOKEN]
                sentence_vector[j] = np.concatenate((np.array([index]), self.char_model.make_one_hot(word.split('#')[0])))
            self.vectors.append(sentence_vector)
        self.lex.close_dicts()
        print('Done in ' + str(time.time() - t) + 's')

