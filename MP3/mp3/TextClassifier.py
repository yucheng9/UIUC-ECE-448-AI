# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from math import log
class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification
        
        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.wordP = {}
        self.typeP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.zero_default = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.zero_default_bi = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.bigram = {}
        self.lambda_mixture = 1

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        # TODO: Write your code here
        emptyP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        types_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        types_words = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        smooth_const = 0.1
        pairs = {}
        pair_num = 0
        for i in range(len(train_set)):
            curr_class = train_label[i] - 1
            types_num[curr_class] += 1
            doc_len = len(train_set[i])
            for j in range(doc_len):
                w = train_set[i][j]
                # unigram
                if w not in self.wordP.keys():
                    self.wordP[w] = []
                    self.wordP[w].extend(emptyP)
                self.wordP[w][curr_class] += 1
                types_words[curr_class] += 1
                
                # bigram
                if w not in self.bigram.keys():
                    self.bigram[w] = {}
                    if w not in pairs:
                        pairs[w] = []
                if (j + 1 < doc_len):
                    w_next = train_set[i][j + 1]
                    if w_next not in self.bigram[w].keys():
                        self.bigram[w][w_next] = []
                        self.bigram[w][w_next].extend(emptyP)
                        if w_next not in pairs[w]:
                            pairs[w].append(w_next)
                            pair_num += 1
                    self.bigram[w][w_next][curr_class] += 1
                        
                
        all_words = self.wordP.keys()
        word_num = len(all_words)
        for i in range(14):
            self.typeP[i] = types_num[i] / sum(types_num)
            self.zero_default[i] = smooth_const/(types_words[i] + smooth_const * word_num)
            self.zero_default_bi[i] = smooth_const/(types_words[i] - types_num[i] + smooth_const * pair_num)
        for word in all_words:
            for i in range(len(self.wordP[word])):
                self.wordP[word][i] = (self.wordP[word][i] + smooth_const)/(types_words[i] + smooth_const * word_num)
            for w_next in self.bigram[word]:
                for i in range(len(self.bigram[word][w_next])):
                    self.bigram[word][w_next][i] = (self.bigram[word][w_next][i] + smooth_const)/(types_words[i] - types_num[i] + smooth_const * pair_num)
               
        print('Top 20 feature words')
        for type_idx in range(14):
            print('Class', type_idx)
            curr_top_20 = []
            for word in all_words:
                prob = self.wordP[word][type_idx]
                if len(curr_top_20) == 0:
                    curr_top_20.append(word)
                else:
                    for i in range(len(curr_top_20)):
                        if prob > self.wordP[curr_top_20[i]][type_idx]:
                            curr_top_20.insert(i, word)
                            break
                if (len(curr_top_20) > 20):
                    curr_top_20.pop()
            print(curr_top_20)
        print('')
        

    def predict(self, x_set, dev_label,lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """

        accuracy = 0.0
        result = []

        # TODO: Write your code here
        for k in range(len(x_set)):
            doc = x_set[k]
            max_result = 0
            max_prob = 0
            for i in range(14):
                if (doc[0] in self.wordP):
                    curr_prob = log(self.typeP[i]) + log(self.wordP[doc[0]][i])
                    curr_prob_bi = log(self.typeP[i]) + log(self.wordP[doc[0]][i])
                else:
                    curr_prob = log(self.typeP[i]) + log(self.zero_default[i])
                    curr_prob_bi = log(self.typeP[i]) + log(self.zero_default[i])
                for h in range(len(doc) - 1):
                    word = doc[h]
                    w_next = doc[h + 1]
                    if (w_next in self.wordP):
                        curr_prob += log(self.wordP[w_next][i])
                    else:
                        curr_prob += log(self.zero_default[i])
                    if (word in self.bigram):
                        if (w_next in self.bigram[word]):
                            curr_prob_bi += log(self.bigram[word][w_next][i])
                        else:
                            curr_prob_bi += log(self.zero_default_bi[i])
                    else:
                        curr_prob_bi += log(self.zero_default_bi[i])
                curr_prob = curr_prob * (1 - self.lambda_mixture) + curr_prob_bi * self.lambda_mixture
                    
                if (i == 0 or curr_prob > max_prob):
                    max_prob = curr_prob
                    max_result = i
            result.append(max_result + 1)
            if (max_result + 1 == dev_label[k]):
                accuracy += 1
                
        accuracy = accuracy/len(dev_label)

        return accuracy,result

