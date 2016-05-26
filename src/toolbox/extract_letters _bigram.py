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

label_dict = {'!': 1, '#' : 2, '$' : 3, '&': 4, '*': 5, ',': 6, '-' : 7, '.': 8, '4' : 9, ':' : 10, ';' : 11, '?' : 12,
              '\\' : 13, '^': 14, 'a': 15, 'b': 16, 'c': 17, 'd' : 18, 'e' : 19, 'f' : 20, 'g' : 21, 'h' : 22, 'i' : 23,
              'k' : 24, 'l' : 25, 'm' : 26, 'n' : 27, 'o' : 28, 'p' : 29, 'q' : 30, 'r' : 31, 's' : 32, 't' : 33,
              'u' : 34, 'v' : 35, 'w' : 36, 'x' : 37, 'y' : 38, 'z' : 39}

word_log = {}


def extractWords(wordfile):
    lines, _ = wordio.read(wordfile)
    assert pth.exists('../crops/')
    word_array = []
    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line):
            temp_word = ""
            for char_idx, char in enumerate(word.characters):
                if (re.match("^[a-zA-Z]", char.text)):
                    char.text = char.text.decode('utf-8', 'ignore').encode("utf-8").lower()
                    temp_word = temp_word + char.text
        word_array.append(temp_word)

    return word_array


def extractImages(wordfile, imgfile, l_file, monogram, bigram, trigram):
    print wordfile
    lines, _ = wordio.read(wordfile)
    img = Image.open(imgfile)
    #img = pamImage.PamImage(imgfile)

    #line_iter = iter(lines)
    #cur_line = line_iter.next()
    #word_iter = iter(cur_line)

    assert pth.exists('../crops/')
    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line):
            previous = ""
            second_previous = ""
            for idx, char in enumerate(word.characters):
                char.text = char.text.decode('utf-8', 'ignore').encode("utf-8").lower()
                if(re.match("^[a-zA-Z]",char.text)):
                    if char.text in monogram:
                        monogram[char.text] += 1
                    else:
                        monogram[char.text] = 1

                    if idx>0 and previous!="":
                        if char.text+previous in bigram:
                            bigram[char.text+previous] +=1
                        else:
                            bigram[char.text+previous] = 1

                        if idx>1 and second_previous!="":
                            if char.text+previous+second_previous in trigram:
                                trigram[char.text + previous + second_previous] += 1
                            else:
                                trigram[char.text+previous+second_previous] = 1
                        else:
                            previous = char.text
                            second_previous = previous
                    else:
                        previous = char.text
                else:
                    previous = ""
                    second_previous = ""

    return monogram,bigram,trigram


if __name__ == "__main__":

    label_file = open('labels.txt', 'w')
    label_file.write('# label file\n')

    monogram = {}
    bigram = {}
    trigram = {}
    word_array = []
    print 'Cropping...'
    assert pth.exists('../charannotations/KNMP') and pth.exists('../charannotations/Stanford')

    for f in listdir('../charannotations/KNMP'):
        wordfile = '../charannotations/KNMP/' + f
        words = extractWords(wordfile)
        word_array.extend(words)

    for f in listdir('../charannotations/Stanford'):
        wordfile = '../charannotations/Stanford/' + f
        word_array.extend(extractWords(wordfile))


    word_array= list(set(word_array))
    print word_array
    with open('vocabulary.pickle', 'wb') as handle:
        pickle.dump(word_array, handle)




    """
    for f in listdir('../charannotations/KNMP'):
        wordfile = '../charannotations/KNMP/' + f
        imgfile = wordfile.replace('.words', IM_EXT).replace('/charannotations', '/pages').replace('2C20', '2C2O')

        monogram,bigram,trigram= extractImages(wordfile, imgfile, label_file,monogram,bigram,trigram)

    for f in listdir('../charannotations/Stanford'):
        wordfile = '../charannotations/Stanford/' + f
        imgfile = wordfile.replace('.words', IM_EXT).replace('/charannotations', '/pages').replace('2C20', '2C2O')

        monogram, bigram, trigram= extractImages(wordfile, imgfile, label_file,monogram,bigram,trigram)

    total_monogram = sum(monogram.values())
    total_bigram = sum(bigram.values())
    total_trigram = sum(trigram.values())
    monogram.update((x,float(y)/total_monogram) for x,y in monogram.items())
    bigram.update((x,float(y)/total_bigram) for x,y in bigram.items())

    trigram.update((x,float(y)/total_trigram) for x,y in trigram.items())
    #monogram =  sorted(monogram.items(), key=operator.iemgetter(1))
    #bigram = sorted(bigram.items(),key=operator.itemgetter(1))
    #trigram = sorted(trigram.items(),key=operator.itemgetter(1))
    print 'monogram:' + str(monogram)
    print 'bigram:' + str(bigram)
    print 'trigram:' + str(trigram)
    with open('monogram.pickle', 'wb') as handle:
        pickle.dump(monogram, handle)
    with open('bigram.pickle', 'wb') as handle:
        pickle.dump(bigram, handle)
    with open('trigram.pickle', 'wb') as handle:
        pickle.dump(trigram, handle)"""
