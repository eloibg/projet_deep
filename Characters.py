import numpy as np


class Characters:
    def __init__(self):
        self.char_dict = {'<UNK>': 0}
        self.n_words = 1

    def add_char(self, word):
        for char in word:
            if char in self.char_dict:
                continue
            else:
                self.char_dict[char] = self.n_words
                self.n_words += 1

    def make_one_hot(self, word):
        one_hot_word = np.zeros(self.n_words)
        for i in range(0,len(word)):
            char = word[i]
            one_hot_char = np.zeros(self.n_words)
            try:
                one_hot_char[self.char_dict[char]] = 1
            except:
                one_hot_char[0] = 1
            one_hot_word = np.add(one_hot_word, one_hot_char)
        maximum = np.max(one_hot_word)
        return np.divide(one_hot_word, max(1, maximum))
