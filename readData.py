### Imports + lecture de fichier

from torch import nn
from torch.autograd import Variable
import torch
import csv
import numpy as np
import pandas as pd
import nltk
from Utils import assignSWNTags, check_for_idioms, adjust_idioms_tags

glove = "C:\\Users\\eloib\\Downloads\\glove.twitter.27B\\glove.twitter.27B.100d.txt"
fasttext = "C:\\Users\\eloib\\Downloads\\crawl-300d-2M.vec\\crawl-300d-2M.vec"
train = pd.read_csv("C:\\Users\\eloib\\Downloads\\train\\train.csv")

X_train = train["comment_text"].values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

X_train = X_train[0:20]

fasttext_embeds = pd.read_table(fasttext, sep=" ", index_col=0, header=None, usecols=range(0,301), skiprows=1, quoting=csv.QUOTE_NONE)

print(fasttext_embeds.loc["hiadg"].as_matrix())

### Raw text processing
processed_text = []
for i in range(0, len(X_train)):
    sentence = X_train[i]
    sentence = check_for_idioms(sentence)
    tokens = nltk.wordpunct_tokenize(sentence)
    tokens = nltk.pos_tag(tokens)
    tokens = assignSWNTags(tokens)
    tokens = adjust_idioms_tags(tokens)
    processed_text.append(tokens)

word_list = nltk.Text([item.split('#')[0] for sublist in processed_text for item in sublist]).vocab()
embedding_matrix = np.zeros((len(word_list), 300))
dictionary = {}
i = 0
for word in word_list.keys():
    dictionary[word.split('#')[0]] = i
    i += 1
    #embedding_matrix[0] =
