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
            self.word_to_index[c] = self.n_words
            self.word_to_count[c] = 1
            self.index_to_word[self.n_words] = c
            self.n_words += 1

    def add_sentence(self, sentence):
        for word in list(sentence):
            self.add_word(word)

    def make_one_hot(self, word):
        if word in self.word_to_index:
            one_hot = [0] * self.n_words
            one_hot[self.word_to_index[word]] = 1
            return one_hot


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
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