import pandas as pd
import ast
import json
import csv
import numpy as np
from random import uniform

embed_size = 100
UNK = []
for j in range(0, embed_size):
    UNK.append(uniform(-1.0, 1.0))

glove_data_file = "C:\\Users\\eloib\\Downloads\\glove.twitter.27B\\glove.twitter.27B.100d.txt"
#glove_data_file = "C:\\Users\\eloib\\Downloads\\glove.840B.300d\\glove.840B.300d.txt"
words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

embed_size = 100
UNK = []
for j in range(0, embed_size):
    UNK.append(uniform(-1.0, 1.0))


path_dataset = "C:\\Users\\eloib\\Documents\\Maitrise\\NegDetector\\preprocessedPerspective.csv"
path_sentiment = "C:\\Users\\eloib\\Documents\\Maitrise\\NegDetector\\FinalTests\\perspectiveAllWords"

finn = json.load(open(path_sentiment + "finn.json"))
inquirer = json.load(open(path_sentiment + "inquirer.json"))
liu = json.load(open(path_sentiment + "liu.json"))
nrc = json.load(open(path_sentiment + "nrc.json"))
subj = json.load(open(path_sentiment + "subj.json"))
swn = json.load(open(path_sentiment + "SWN.json"))

data = pd.read_csv(path_dataset, sep='\t', encoding='utf-8', header=None)
sentences = data[1]
scores = data[3]

formatted_sentences = []
augmented_embedding = []

for i in range(0, 1000):
    sentence = ast.literal_eval(sentences[i])
    formatted_sentence = []
    sentence_augmented_embedding = []
    sentence_pos = []
    for word in sentence:
        word_pos = word.split('#')
        formatted_sentence.append(word_pos[0])
        if word_pos[1] == 'n':
            sentence_pos.append(-1)
        elif word_pos[1] == 'v':
            sentence_pos.append(-0.5)
        elif word_pos[1] == 'r':
            sentence_pos.append(0.5)
        elif word_pos[1] == 'a':
            sentence_pos.append(1)
        else:
            sentence_pos.append(0.0)
    for k in range(0, len(formatted_sentence)):
        word = formatted_sentence[k]
        try:
            embed = words.loc[word].as_matrix()
        except:
            embed = UNK
        word_augmented_embedding = embed
        word_augmented_embedding = np.append(word_augmented_embedding, [finn[str(i)][0][k], inquirer[str(i)][0][k], liu[str(i)][0][k],
                                             nrc[str(i)][0][k], subj[str(i)][0][k], swn[str(i)][0][k],
                                             finn[str(i)][1][k], inquirer[str(i)][1][k], liu[str(i)][1][k],
                                             nrc[str(i)][1][k], subj[str(i)][1][k], swn[str(i)][1][k], sentence_pos[k]])
        sentence_augmented_embedding.append(word_augmented_embedding)
    formatted_sentences.append(formatted_sentence)
    augmented_embedding.append(sentence_augmented_embedding)
    print(i)

    if i % 1000 == 0:
        pd.DataFrame(augmented_embedding).to_csv('essai' + str(i) + '.csv', sep='\t', encoding='utf-8', header=None)

pd.DataFrame(augmented_embedding).to_csv('essai' + str(i) + '.csv', sep='\t', encoding='utf-8', header=None)

with open('sentences.txt', 'w+', encoding='utf-8') as f:
    for sentence in formatted_sentences:
        f.write(' '.join(sentence))
        f.write('\n')

with open('scores.txt', 'w+') as f:
    for score in scores:
        f.write(str(score))
        f.write('\n')


