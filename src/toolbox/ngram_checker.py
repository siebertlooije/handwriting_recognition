import gtk
import pamImage
import croplib
import wordio
import word as wrd
import os.path as pth
import re
from os import listdir
import operator
import pickle

IM_EXT='.ppm'

import re
import numpy

from skimage.util import img_as_ubyte
from PIL import Image

import numpy as np
import nltk
import distance
from nltk.util import ngrams


def load_vocabulary():
    vocabulary = []
    with open('vocabulary.pickle', 'r') as handle:
        vocabulary.extend(pickle.load(handle));

    return vocabulary

def load_ngrams():
    dataset_brown = []
    dataset_old = []
    dataset_new = []
    with open('bigram_brown.pickle', 'r') as handle:
        dataset_brown.append(pickle.load(handle))
    with open('trigram_brown.pickle', 'r') as handle:
       dataset_brown.append(pickle.load(handle))
    with open('fourgram_brown.pickle', 'r') as handle:
        dataset_brown.append(pickle.load(handle))
    with open('bigram_old.pickle', 'r') as handle:
        dataset_old.append(pickle.load(handle))
    with open('trigram_old.pickle', 'r') as handle:
        dataset_old.append(pickle.load(handle))
    with open('fourgram_old.pickle', 'r') as handle:
        dataset_old.append(pickle.load(handle))
    with open('bigram.pickle', 'r') as handle:
        dataset_new.append(pickle.load(handle))
    with open('trigram.pickle', 'r') as handle:
        dataset_new.append(pickle.load(handle))
    dataset = {}
    dataset['brown'] = dataset_brown
    dataset['old'] = dataset_old
    dataset['new'] = dataset_new
    return dataset

#https://en.wikipedia.org/wiki/Jaccard_index
#https://nl.wikipedia.org/wiki/Levenshteinafstand
def calculateDistance(word_array,vocabulary):
    distance_score_min_levenshstein = {}
    distance_score_sum_levenshstein = {}

    for voc in vocabulary:
        temp_score_lev = 9999
        mean_score = 0
        for word in word_array:
            score = distance.levenshtein(word,voc);
            if temp_score_lev > score:
                temp_score_lev = score
        distance_score_min_levenshstein[voc] = temp_score_lev
    return distance_score_min_levenshstein


def extractNGrams(words):
    trigrams_save = {}
    bigrams_save = {}
    fourgrams_save = {}
    whitelist = set('abcdefghijklmnopqrstuvwxy')
    for word in words:
        word = word.decode('utf-8', 'ignore').encode("utf-8").lower()

        word = ''.join(filter(whitelist.__contains__, word))
        chrs = [c for c in word]

        trigrams = ngrams(chrs, 3)
        for gram in trigrams:
            gram = gram[0] + gram[1] + gram[2]
            if gram in trigrams_save:
                trigrams_save[gram] += 1
            else:
                trigrams_save[gram] = 1

        bigrams = ngrams(chrs, 2)
        for gram in bigrams:
            gram = gram[0] + gram[1]
            if gram in bigrams_save:
                bigrams_save[gram] += 1
            else:
                bigrams_save[gram] = 1

        fourgrams = ngrams(chrs, 4)
        for gram in fourgrams:
            gram = gram[0] + gram[1] + gram[2] + gram[3]
            if gram in fourgrams_save:
                fourgrams_save[gram] += 1
            else:
                fourgrams_save[gram] = 1

    total_bigram_save = sum(bigrams_save.values())
    total_trigram_save = sum(trigrams_save.values())
    total_fourgram_save = sum(fourgrams_save.values())
    bigrams_save.update((x, float(y) / total_bigram_save) for x, y in bigrams_save.items())
    trigrams_save.update((x, float(y) / total_trigram_save) for x, y in trigrams_save.items())
    fourgrams_save.update((x, float(y) / total_fourgram_save) for x, y in fourgrams_save.items())
    ngram ={}


    ngram['fourgram'] = fourgrams_save
    ngram['trigram'] = trigrams_save
    ngram['bigram'] = bigrams_save
    return ngram

def checkWordInNgrams2(word_array, dataset):
    ngram_key = ['bigram','trigram','fourgram']
    words_exist = list(word_array)

    for word in word_array:
        if len(word) < 2:
            continue 
        word_removed = False
        for key in ngram_key:    
            value_dict = dataset[key].iterkeys().next()
            n = len(value_dict)
            ngram_array = [word[j:j + n] for j in range(0, len(word) - 1, 1)]
            for ngram in ngram_array:
                if len(ngram) == n: #to be removed
                    if ngram not in dataset[key]:
                        words_exist.remove(word)
                        word_removed = True
                        break
            if word_removed :
                break

    return words_exist

def checkWordInNgrams(word_array, dataset):
    ngram_key = ['bigram','trigram','fourgram']
    words_exist = []
    for word in word_array:
        length_word = len(word)
        for key in ngram_key:
                if length_word < 2:
                    return words_exist
                else:
                    value_dict = dataset[key].iterkeys().next()
                    n = len(value_dict)
                    ngram_array = [word[j:j + n] for j in range(0, len(word) - 1, 1)]
                    for ngram in ngram_array:

                        if len(ngram) == n:
                            if ngram in dataset[key]:
                                #print "dataset:  " + str(key) + " word:" + word + 'with ngram: ' + str(ngram)
                                if word not in words_exist:
                                    words_exist.append(word)
                            else:
                                if word in words_exist:
                                    words_exist.remove(word)
    return words_exist

if __name__ == "__main__":
    print_state = True
    vocabulary = load_vocabulary()
    word_array = ["que","quei","qiai","qiie","qiiei","cnai","cnie","cniei","caue","cauei","caiai","caiie","caiiei"]
    #word_array = ["que"]
    #dataset = load_ngrams()
    #input_word = "zxllo"
    #checkWordInNgrams(input_word,dataset)
    ngram_voc= extractNGrams(vocabulary)
    word_exists = checkWordInNgrams(word_array,ngram_voc)

    if len(word_exists) == 1:
        print "word: " + word_exists[0] +  "   because it only exists in vocabulary"
    else:
        lev = calculateDistance(word_array, vocabulary)
        lev = sorted(lev.items(), key=operator.itemgetter(1))
        for k, v in lev:
            print "word:"+  k + "   with score:" + str(v)

    # ngram_word = extractNGrams(word_exists)

    #lev= calculateDistance(word_array,vocabulary)
    #lev = sorted(lev.items(), key=operator.itemgetter(1))
    #counter = 0;
    #print "With word array : " + str(word_array)
    #print "With word exists array :" + str(word_exists)
    #print "With vocabulary :" + str(vocabulary)
    """
    if(print_state):
        for key,value in lev:
            if counter ==0:
                print "Best levenstein:" + str(key)+ "   distance score:" + str(value)
            else:
                print "Other levenstein:" + str(key) + "   distance score:" + str(value)
            counter += 1

    best_words = {}
    #The score of the best word
    best_score = lev[0][1]
    for key, value in lev:
        if value <= best_score + best_score*0.2:
            best_words[key] = value;
        else:
            break;

    print best_words
    """


