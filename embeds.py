import pandas as pd
import csv

glove_data_file = "C:\\Users\\eloib\\Downloads\\glove.twitter.27B\\glove.twitter.27B.200d.txt"

words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def vec(w):
  return words.loc[w].as_matrix()