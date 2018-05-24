import random

CHAR_LIST = 'abcdefghijklmnopqrstuvwxyz1234567890.,\';:^()#<>[]{}!"/$%?&*'

def generate_random_subversion(word):
    new_word = ""
    for char in word:
        random_int = random.randint(1, 100)
        new_char = char
        if random_int > 75:
            random_int = random.randint(0, 58)
            new_char = CHAR_LIST[random_int]
        new_word += new_char
    return new_word

def generate_bad_word_list():
    bad_word_list = []
    with open('popularBadWords.txt') as bad_words:
        for line in bad_words.readlines():
            bad_word_list.append(line.split('     ')[0])
    return bad_word_list

def add_noise(sentence):
    bad_word_list = generate_bad_word_list()
    new_sent = []
    for word in sentence:
        if word in bad_word_list:
            new_word = generate_random_subversion(word)
            new_sent.append(new_word)
        else:
            new_sent.append(word)
    return new_sent