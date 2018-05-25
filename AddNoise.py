import random

CHAR_LIST = 'abcdefghijklmnopqrstuvwxyz1234567890.,\';:^()#<>[]{}!"/$%?&*'

def new_char_generator():
    return CHAR_LIST[random.randint(0, 58)]

def generate_random_subversion(word):
    rand_ints = []
    if len(word) > 5:
        rand_ints.append(random.randint(0, 5))
        rand_ints.append(random.randint(5, len(word)-1))
    else:
        rand_ints.append(random.randint(0, len(word)-1))
    new_word = ""
    for i, char in enumerate(word):
        if i in rand_ints:
            new_char = new_char_generator()
        else:
            new_char = char
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