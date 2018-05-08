from Utils import negationCheck
import json

class LexiconClassify:
    def __init__(self, data):
        self.data = data
        self.sentiment_scores = []

    def open_dicts(self):
        self.pos_finn = json.load(open('JSONDicts\\finn' + 'PosDict.json'))
        self.neg_finn = json.load(open('JSONDicts\\finn' + 'NegDict.json'))
        self.pos_inquirer = json.load(open('JSONDicts\\inquirer' + 'PosDict.json'))
        self.neg_inquirer = json.load(open('JSONDicts\\inquirer' + 'NegDict.json'))
        self.pos_liu = json.load(open('JSONDicts\\liu' + 'PosDict.json'))
        self.neg_liu = json.load(open('JSONDicts\\liu' + 'NegDict.json'))
        self.pos_nrc = json.load(open('JSONDicts\\nrc' + 'PosDict.json'))
        self.neg_nrc = json.load(open('JSONDicts\\nrc' + 'NegDict.json'))
        self.pos_subj = json.load(open('JSONDicts\\subj' + 'PosDict.json'))
        self.neg_subj = json.load(open('JSONDicts\\subj' + 'NegDict.json'))
        self.pos_SWN = json.load(open('JSONDicts\\SWN' + 'PosDict.json'))
        self.neg_SWN = json.load(open('JSONDicts\\SWN' + 'NegDict.json'))
        self.dict_list = [(self.pos_finn, self.neg_finn), (self.pos_inquirer, self.neg_inquirer),
                          (self.pos_liu, self.neg_liu),
                          (self.pos_nrc, self.neg_nrc), (self.pos_subj, self.neg_subj), (self.pos_SWN, self.neg_SWN)]

    def close_dicts(self):
        self.pos_finn = None
        self.neg_finn = None
        self.pos_inquirer = None
        self.neg_inquirer = None
        self.pos_liu = None
        self.neg_liu = None
        self.pos_nrc = None
        self.neg_nrc = None
        self.pos_subj = None
        self.neg_subj = None
        self.pos_SWN = None
        self.neg_SWN = None
        self.dict_list = None

    def classify(self):
        self.open_dicts()
        for i in range(0, len(self.data)):
            word_pos_values = []
            word_neg_values = []
            for dict in self.dict_list:
                posDict = dict[0]
                negDict = dict[1]
                pos_list = []
                neg_list = []
                for j in range(0, len(self.data[i])):
                    word = self.data[i][j]
                    if word in posDict:
                        pos_value = posDict[word]
                        neg_value = negDict[word]
                        if (pos_value > 0 or neg_value > 0) and \
                                negationCheck(word, self.data[i][max(0, j-4):min(len(self.data[i]), j+3)]):
                            pos_list.append(neg_value)
                            neg_list.append(pos_value)
                        else:
                            pos_list.append(pos_value)
                            neg_list.append(neg_value)
                    else:
                        neg_list.append(0)
                        pos_list.append(0)
                word_pos_values.append(pos_list)
                word_neg_values.append(neg_list)
            sentiment_per_word = []
            for k in range(0, len(word_pos_values[0])):
                dict_score_list = []
                for l in range(0, len(self.dict_list)):
                    dict_score_list.append(word_pos_values[l][k])
                    dict_score_list.append(word_neg_values[l][k])
                sentiment_per_word.append(dict_score_list)
            self.sentiment_scores.append(sentiment_per_word)
        self.close_dicts()
