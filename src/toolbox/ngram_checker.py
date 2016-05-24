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

def checkWordInNgrams(input_word,dataset):
    length_word = len(input_word)
    for key in dataset:
        for i in range(0,len(dataset[key])):

            if length_word < 2:
                return;
            else:
                n = 0
                if i == 0:
                    n = 2
                elif i == 1:
                    n = 3
                else:
                    n = 4

                ngram_array = [input_word[j:j + n] for j in range(0, len(input_word) - 1, 1)]
                print ngram_array
                for ngram in ngram_array:
                    if len(ngram) == n:
                        if ngram in dataset[key][n-2]:
                            print "dataset:  " + str(key) + "  has with i :" + str(i) + " word:" + input_word + 'with ngram: ' + str(ngram)
                            break;


if __name__ == "__main__":
    dataset = load_ngrams()
    input_word = "zxllo"
    checkWordInNgrams(input_word,dataset)

