"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from math import log

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    
    data = {}
    count = {}
    tags = []
    tt = ''
    for sentence in train:
        for word in sentence:
            if data.get(word[1]) is None:
                data[word[1]] = {}
                count[word[1]] = 0
                tags.append(word[1])
            count[word[1]] += 1
            if data[word[1]].get(word[0]) is None:
                data[word[1]][word[0]] = 1
            else:
                data[word[1]][word[0]] += 1
    tt = max(count, key=count.get)

    predicts = []
    for i in range(len(test)):
        predicts.append([])
        for word in test[i]:
            predict = tt
            count1 = -1
            for t in tags:
                if data[t].get(word) is not None:
                    if data[t].get(word) > count1:
                        count1 = data[t].get(word)
                        predict = t
            predicts[i].append((word, predict))
    
    return predicts


def viterbi_p1(train, test):
    '''
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    predicts = []
    initial_t_n = {}
    sen_num = 0
    tag_pair = {}
    tag_word = {}
    hapax_words = {}
    #initial words
    for i in range(len(train)):
        sen_num += 1
        if train[i][0][1] in initial_t_n:
            initial_t_n[train[i][0][1]] += 1
        else:
            initial_t_n[train[i][0][1]] = 1
        if train[i][0][1] in tag_word:
            if train[i][0][0] in tag_word[train[i][0][1]]:
                tag_word[train[i][0][1]][train[i][0][0]] += 1
            else:
                tag_word[train[i][0][1]][train[i][0][0]] = 1
        else:
            tag_word[train[i][0][1]] = {}
            tag_word[train[i][0][1]][train[i][0][0]] = 1
        if train[i][0][0] in hapax_words:
            hapax_words[train[i][0][0]] = 'multiple'
        else:
            hapax_words[train[i][0][0]] = train[i][0][1]
            
    #initial probability      
    initial_p = []
    initial_t = []
    for tag in initial_t_n:
        i = 0
        count_p = len(initial_p)
        if count_p > 0:
            while (i < count_p):
                if initial_t_n[tag] > initial_p[i]:
                    initial_p.insert(i, initial_t_n[tag])
                    initial_t.insert(i, tag)
                    break
                i += 1
                if i == count_p:
                    initial_p.insert(i, initial_t_n[tag])
                    initial_t.insert(i, tag)
        else:
            initial_p.append(initial_t_n[tag])
            initial_t.append(tag)
     
    #all words
    for i in range(len(train)):
        for j in range(1,len(train[i])):
            if train[i][j - 1][1] in tag_pair:
                if train[i][j][1] in tag_pair[train[i][j - 1][1]]:
                    tag_pair[train[i][j - 1][1]][train[i][j][1]] += 1
                else:
                    tag_pair[train[i][j - 1][1]][train[i][j][1]] = 1
            else:
                tag_pair[train[i][j - 1][1]] = {}
                tag_pair[train[i][j - 1][1]][train[i][j][1]] = 1
            
            if train[i][j][1] in tag_word:
                if train[i][j][0] in tag_word[train[i][j][1]]:
                    tag_word[train[i][j][1]][train[i][j][0]] += 1
                else:
                    tag_word[train[i][j][1]][train[i][j][0]] = 1
            else:
                tag_word[train[i][j][1]] = {}
                tag_word[train[i][j][1]][train[i][j][0]] = 1
            
            if train[i][j][0] in hapax_words:
                hapax_words[train[i][j][0]] = 'multiple'
            else:
                hapax_words[train[i][j][0]] = train[i][0][1]
        
    for tag in tag_word:
        k = 0.0001
        for word in hapax_words:
            if word in tag_word[tag]:
                tag_word[tag][word] += k
            else:
                tag_word[tag][word] = k
        tag_word[tag]['unseen'] = k
        
    for tag in tag_pair:
        for sec_tag in tag_word:
            if sec_tag not in tag_pair[tag]:
                tag_pair[tag][sec_tag] = 1
            else:
                tag_pair[tag][sec_tag] += 1
            
    
    for sen in test:
        word_p = []
        sen_predict = []
        for i in range(len(sen)):
            word = sen[i]
            tag_max = [float('-inf'), '2222']
            if i == 0:
                cur_p = {}
                for tag in initial_t_n:
                    cur_p[tag] = [log(initial_t_n[tag]) + log(tag_word[tag].get(word, tag_word[tag]["unseen"])), "1111"]
                    if cur_p[tag][0] > tag_max[0]:
                        tag_max = [cur_p[tag][0], tag]
                word_p.append(cur_p)
                
            else:
                cur_p = {}
                
                for tag in tag_word:
                    cur_max = [float('-inf'), '2222']
                    for pre_tag in word_p[i - 1]:
                        cur_tag_p = word_p[i - 1][pre_tag][0] + log(tag_pair[pre_tag][tag]) + log(tag_word[tag].get(word, tag_word[tag]['unseen']))
                        if cur_tag_p > cur_max[0]:
                            cur_max = [cur_tag_p, pre_tag]
                    cur_p[tag] = cur_max
                    if cur_max[0] > tag_max[0]:
                        tag_max = [cur_max[0], tag]
                word_p.append(cur_p)
            #sen_predict.append([sen[i], tag_max[1]])
        
        
        pre_tag = '3333'
        for i in range(len(sen) - 1, -1, -1):
            cur_re = []
            if i == len(sen) - 1:
                cur_max = [float('-inf'), ['none', 'none']]
                for tag in word_p[i]:
                    if (word_p[i][tag][0] > cur_max[0]):
                        cur_max = [word_p[i][tag][0], [sen[i], tag]]
                        pre_tag = word_p[i][tag][1]
                cur_re = cur_max[1]
            else:
                cur_re = [sen[i], pre_tag]
                if i > 0:
                    pre_tag = word_p[i][pre_tag][1]
            sen_predict.insert(0, cur_re)
        
        predicts.append(sen_predict)
        
    return predicts

def viterbi_p2(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''


    predicts = []
    initial_t_n = {}
    sen_num = 0
    tag_pair = {}
    tag_word = {}
    hapax_words = {}
    hapax_words_num = 0
    #initial words
    for i in range(len(train)):
        sen_num += 1
        if train[i][0][1] in initial_t_n:
            initial_t_n[train[i][0][1]] += 1
        else:
            initial_t_n[train[i][0][1]] = 1
        if train[i][0][1] in tag_word:
            if train[i][0][0] in tag_word[train[i][0][1]]:
                tag_word[train[i][0][1]][train[i][0][0]] += 1
            else:
                tag_word[train[i][0][1]][train[i][0][0]] = 1
        else:
            tag_word[train[i][0][1]] = {}
            tag_word[train[i][0][1]][train[i][0][0]] = 1
        if train[i][0][0] in hapax_words:
            hapax_words[train[i][0][0]] = 'multiple'
        else:
            hapax_words[train[i][0][0]] = train[i][0][1]
            
    #initial probability      
    initial_p = []
    initial_t = []
    for tag in initial_t_n:
        i = 0
        count_p = len(initial_p)
        if count_p > 0:
            while (i < count_p):
                if initial_t_n[tag] > initial_p[i]:
                    initial_p.insert(i, initial_t_n[tag])
                    initial_t.insert(i, tag)
                    break
                i += 1
                if i == count_p:
                    initial_p.insert(i, initial_t_n[tag])
                    initial_t.insert(i, tag)
        else:
            initial_p.append(initial_t_n[tag])
            initial_t.append(tag)
     
    #all words
    for i in range(len(train)):
        for j in range(1,len(train[i])):
            if train[i][j - 1][1] in tag_pair:
                if train[i][j][1] in tag_pair[train[i][j - 1][1]]:
                    tag_pair[train[i][j - 1][1]][train[i][j][1]] += 1
                else:
                    tag_pair[train[i][j - 1][1]][train[i][j][1]] = 1
            else:
                tag_pair[train[i][j - 1][1]] = {}
                tag_pair[train[i][j - 1][1]][train[i][j][1]] = 1
            
            if train[i][j][1] in tag_word:
                if train[i][j][0] in tag_word[train[i][j][1]]:
                    tag_word[train[i][j][1]][train[i][j][0]] += 1
                else:
                    tag_word[train[i][j][1]][train[i][j][0]] = 1
            else:
                tag_word[train[i][j][1]] = {}
                tag_word[train[i][j][1]][train[i][j][0]] = 1
            
            if train[i][j][0] in hapax_words:
                hapax_words[train[i][j][0]] = 'multiple'
            else:
                hapax_words[train[i][j][0]] = train[i][0][1]
                
    hapax_tag_word = {}
    for word in hapax_words:
        if hapax_words[word] != 'multiple':
            hapax_words_num += 1
            if hapax_words[word] in hapax_tag_word:
                hapax_tag_word[hapax_words[word]] += 1
            else:
                 hapax_tag_word[hapax_words[word]] = 1
        
    
    for tag in tag_word:
        k = 0.0001 * hapax_tag_word.get(tag, 1)/hapax_words_num
        for word in hapax_words:
            if word in tag_word[tag]:
                tag_word[tag][word] += k
            else:
                tag_word[tag][word] = k
        tag_word[tag]['unseen'] = k
    
    for tag in tag_pair:
        for sec_tag in tag_word:
            if sec_tag not in tag_pair[tag]:
                tag_pair[tag][sec_tag] = 1
            else:
                tag_pair[tag][sec_tag] += 1
            
    
    for sen in test:
        word_p = []
        for i in range(len(sen)):
            word = sen[i]
            if i == 0:
                cur_p = {}
                for tag in initial_t_n:
                    cur_p[tag] = [log(initial_t_n[tag]) + log(tag_word[tag].get(word, tag_word[tag]["unseen"])), "1111"]
                word_p.append(cur_p)
            else:
                cur_p = {}
                for tag in tag_word:
                    cur_max = [float('-inf'), '2222']
                    for pre_tag in word_p[i - 1]:
                        cur_tag_p = word_p[i - 1][pre_tag][0] + log(tag_pair[pre_tag][tag]) + log(tag_word[tag].get(word, tag_word[tag]['unseen']))
                        if cur_tag_p > cur_max[0]:
                            cur_max = [cur_tag_p, pre_tag]
                    cur_p[tag] = cur_max
                word_p.append(cur_p)
        
        sen_predict = []
        pre_tag = '3333'
        for i in range(len(sen) - 1, -1, -1):
            cur_re = []
            if i == len(sen) - 1:
                cur_max = [float('-inf'), ['none', 'none']]
                for tag in word_p[i]:
                    if (word_p[i][tag][0] > cur_max[0]):
                        cur_max = [word_p[i][tag][0], [sen[i], tag]]
                        pre_tag = word_p[i][tag][1]
                cur_re = cur_max[1]
            else:
                cur_re = [sen[i], pre_tag]
                if i > 0:
                    pre_tag = word_p[i][pre_tag][1]
            sen_predict.insert(0, [cur_re[0], cur_re[1]])
        into_pre = sen_predict.copy()
        predicts.append(into_pre)
    return predicts





