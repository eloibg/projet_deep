import pandas as pd
import csv
import numpy as np

glove_data_file = "C:\\Users\\eloib\\Downloads\\glove.twitter.27B\\glove.twitter.27B.200d.txt"

words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def vec(w):
  return words.loc[w].as_matrix()

words_matrix = words.as_matrix()

def find_closest_word(v):
  diff = words_matrix - v
  delta = np.sum(diff * diff, axis=1)
  i = np.argmin(delta)
  return words.iloc[i].name

print(find_closest_word("hey"))