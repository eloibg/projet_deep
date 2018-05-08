from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import re

def classifySentence(type, sentiment, tone):
    if pd.isnull(type):
        type = ""
    if pd.isnull(sentiment):
        sentiment = ""
    if pd.isnull(tone):
        tone = ""
    sarcasmFlag = False
    if "Snarky/humorous" in type or "Sarcastic" in tone or "Funny" in tone:
        sarcasmFlag = True
    if "negative" in sentiment:
        if "Mean" in tone or "Flamewar" in type or ("Argumentative" in type and "Controversial" in tone):
            tag = "clear negative"
        else:
            tag = "slight negative"
    elif "mixed" in sentiment:
        if "Mean" in tone or "Flamewar" in type or ("Argumentative" in type and "Controversial" in tone):
            tag = "slight negative"
        elif "Positive" in type or "Informative" in tone or "Sympathetic" in tone:
            tag = "slight positive"
        else:
            tag = "neutral"
    elif "positive" in sentiment:
        if "Positive" in type or "Informative" in tone or "Sympathetic" in tone:
            tag = "clear positive"
        else:
            tag = "slight positive"
    else:
        tag = "neutral"
    return sarcasmFlag, tag

def SWNTag(POSTag):
    if any(tag in POSTag for tag in("CD", "DT", "JJ")):
        return "a"
    elif any(tag in POSTag for tag in("EX", "IN", "RB")):
        return "r"
    elif any(tag in POSTag for tag in("MD", "VB")):
        return "v"
    elif "NN" in POSTag:
        return "n"
    else:
        return "not tagged"

def assignSWNTags(tags):
    wnl = WordNetLemmatizer()
    newWordList = []
    for word in tags:
        root = word[0].lower()
        tag = SWNTag(word[1])
        #if tag != "not tagged":
        #    lemmatizedWord = wnl.lemmatize(root, tag)
        #else:
        #    lemmatizedWord = wnl.lemmatize(root)
        #newWordList.append(lemmatizedWord+'#'+ tag)
        newWordList.append(root + '#' + tag)
    return newWordList

def negationCheck(word, tagged_words):
    rules = [line.rstrip() for line in open('negex_triggers.txt')]
    sentence = ""
    for tagged_word in tagged_words:
        sentence = sentence + tagged_word.split('#')[0] + " "
    if word.split('#')[0] in rules:
        return False   ### You can negate a negation
    else:
        return any([negation in sentence.split(' ') for negation in rules])

def extractPOSTag(word):
    word = word.split('#')
    return word[1]

def check_for_idioms(sentence):
    sentence = re.sub('[sS]hut [uU]p', 'shut_up', sentence)
    return sentence

def adjust_idioms_tags(words):
    for i in range(0,len(words)):
        if 'shut_up' in words[i]:
            words[i] = 'shut_up#v'
    return words
