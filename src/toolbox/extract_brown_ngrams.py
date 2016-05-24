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

if __name__ == "__main__":
    brown_corpus = False;
    words = []
    if(brown_corpus):
        nltk.download('brown')
        from nltk.corpus import brown
        corpus = brown
    else:
        import os
        nltk.download('punkt')
        for file in os.listdir("/home/siebert/handwriting_recognition/src/old_corpus"):
            if file.endswith(".txt"):
                with open("/home/siebert/handwriting_recognition/src/old_corpus/"+file,"r") as my_file:
                    data=my_file.read().replace('\n', '')
                    tokens = nltk.word_tokenize(data)
                    words.extend(tokens)

   # print 'getting all the words with length:'+str(len(words))
    from nltk.util import ngrams
    trigrams_save = {}
    bigrams_save = {}
    fourgrams_save = {}
    counter = 0
    whitelist = set('abcdefghijklmnopqrstuvwxy')
    #for word in corpus.words():
    for word in words:
        word = word.decode('utf-8', 'ignore').encode("utf-8").lower()
        word = ''.join(filter(whitelist.__contains__, word))
        chrs = [c for c in word]

        trigrams = ngrams(chrs, 3)
        for gram in trigrams:
            gram  = gram[0]+gram[1]+gram[2]
            if gram in trigrams_save:
                trigrams_save[gram] += 1
            else:
                trigrams_save[gram] = 1

        bigrams = ngrams(chrs,2)
        for gram in bigrams:
            gram = gram[0]+gram[1]
            if gram in bigrams_save:
                bigrams_save[gram] += 1
            else:
                bigrams_save[gram] = 1

        fourgrams = ngrams(chrs,4)
        for gram in fourgrams:
            gram = gram[0]+gram[1]+gram[2] + gram[3]
            if gram in fourgrams_save:
                fourgrams_save[gram] += 1
            else:
                fourgrams_save[gram] = 1

    print "bigram:" + str(bigrams_save)
    print "trigram:" + str(trigrams_save)
    print "fourgram:" + str(fourgrams_save)

    print "done!"
    if(brown_corpus):
        with open('bigram_brown.pickle', 'wb') as handle:
            pickle.dump(bigrams_save, handle)
        with open('trigram_brown.pickle', 'wb') as handle:
            pickle.dump(trigrams_save, handle)
        with open('fourgram_brown.pickle', 'wb') as handle:
            pickle.dump(fourgrams_save, handle)
    else:
        with open('bigram_old.pickle', 'wb') as handle:
            pickle.dump(bigrams_save, handle)
        with open('trigram_old.pickle', 'wb') as handle:
            pickle.dump(trigrams_save, handle)
        with open('fourgram_old.pickle', 'wb') as handle:
            pickle.dump(fourgrams_save, handle)


    print "done saving !"