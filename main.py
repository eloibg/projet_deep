from string import ascii_lowercase
import torch.nn as nn


BOW_token = 0
EOW_token = 1


class Language:
    def __init__(self):
        self.word_to_index = {}
        self.word_to_count = {}
        self.index_to_word = {BOW_token: "BOW", EOW_token: "EOW"}
        self.n_words = 2

    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.n_words
            self.word_to_count[word] = 1
            self.index_to_word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_to_count[word] += 1


class WordLevelEncoder(Language):
    def __init__(self):
        super(WordLevelEncoder, self).__init__()

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


class CharacterLevelEncoder(Language):
    def __init__(self):
        super(CharacterLevelEncoder, self).__init__()
        for c in ascii_lowercase:
            self.add_word(c)

    def add_sentence(self, sentence):
        for word in list(sentence):
            self.add_word(word)

    def make_one_hot(self, word):
        if word in self.word_to_index:
            word = word.lower()
            one_hot = [0] * self.n_words
            one_hot[self.word_to_index[word]] = 1
            return one_hot


class CNN(nn.Module):
    def __init__(self, n_layers=50):
        super(CNN, self).__init__()

        self.layers = []
        for i in range(n_layers):
            layer = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.AdaptiveAvgPooling1D(1))
            self.layers.append(layer)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out


if __name__ == '__main__':
    premiere_phrase = "une phrase quelconque"
    deuxieme_phrase = "une autre phrase pas rapport"

    lang = CharacterLevelEncoder()
    lang.add_sentence(premiere_phrase)
    lang.add_sentence(deuxieme_phrase)
    print(lang.n_words)
    print(lang.index_to_word)
    print(lang.make_one_hot("p"))