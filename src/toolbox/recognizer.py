#!/usr/bin/env python

import sys
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers import *
from keras.layers.core import Merge
import wordio

from scipy.signal import argrelextrema
from skimage.filters import threshold_otsu
import h5py

import matplotlib.pyplot as plt
import scipy

from ngram_checker import *

from os import listdir
import copy
import pickle as pkl

label_mapping = pkl.load(open('char_mappings.pkl', 'rb'))

class N_gram(object):

    def __init__(self, im, start, end, prediction=None, confidence=None, options = None):
        self.im = im
        self.start = start
        self.end = end

        self.prediction = prediction
        self.confidence = confidence
        self.options = options

    def set_prediction(self, pred):
        self.prediction = pred

    def set_confidence(self, conf):
        self.confidence = conf

    def set_options(self, opts):
        self.options = opts

    def combine(self, other):
        return N_gram(self.im, self.start, other.end,
                      self.prediction + other.prediction, self.confidence + other.confidence)

    def get_confidence(self):
        return self.confidence / len(self.prediction)

    def get_options(self):
        return self.options

    def follows(self, other):
        return self.start == other.end

    def followed_by(self, other):
        return other.start == self.end

def get_monograms_otsu(img, cropped):
    otsu_img = img_as_ubyte(np.copy(np.asarray(cropped.convert('L'))))

    try:
        threshold_global_otsu = threshold_otsu(otsu_img)
    except TypeError:
        print 'Something weird happened'
        return None, None, None
    global_otsu = np.array(otsu_img >= threshold_global_otsu).astype(np.int64)

    hist = np.zeros(global_otsu.shape[1])
    for col in range(global_otsu.shape[1]):
        max_white = 0
        white = 0
        for row in range(global_otsu.shape[0]):
            white += 1 if global_otsu[row, col] == 1 else 0
            max_white = max(white, max_white)
        hist[col] = max_white
    hist = np.convolve(hist, 5 * [1 / 5.], 'same') / 25

    for idx in range(1, len(hist) - 1):
        if hist[idx] == hist[idx + 1] and hist[idx - 1] < hist[idx]:
            hist[idx] = (hist[idx + 1] + hist[idx - 1]) / 2

    maxes = argrelextrema(hist, np.greater)
    cuts = maxes[0]
    monograms = []

    for idx, cut in enumerate(cuts):
        for next in cuts[idx + 1:]:
            if 10 < next - cut < 60:
                monograms.append(N_gram(img[:, cut:next], cut, next))

    return monograms, cuts[0], cuts[-1]

def get_monograms(img):
    hist = np.mean(img, axis=(0, 2))

    hist = np.maximum(np.mean(img[img.shape[0] / 2:], axis=(0, 2)), hist)
    hist = np.maximum(np.mean(img[:img.shape[0] / 2], axis=(0, 2)), hist)

    plt.imshow(img)
    smooth_win = 20

    hist_small = np.convolve(hist, 5 * [1 / 5.], 'same') / 25

    hist = np.convolve(hist, 11 * [1 / 11.], 'same') / 25

    hist[:smooth_win - 1] = hist_small[:smooth_win - 1]
    hist[-(smooth_win - 1):] = hist_small[-(smooth_win - 1):]
    maxes = argrelextrema(hist, np.greater)

    cuts = maxes[0]
    for idx in range(len(cuts) - 1):
        while len(cuts) > idx + 1 and cuts[idx + 1] - cuts[idx] < 10:
            cuts[idx] = cuts[idx] if hist[cuts[idx]] > hist[cuts[idx + 1]] else cuts[idx]
            cuts = np.delete(cuts, idx + 1)

    monograms = []

    for idx, cut in enumerate(cuts):
        for next in cuts[idx + 1:]:
            print next, cut
            if 10 < next - cut < 50:
                monograms.append(N_gram(img[:, cut:next], cut, next))

    return monograms, cuts[0], cuts[-1]



def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))   #0
    model.add(Convolution2D(64, 3, 3, activation='relu'))       #1  -1
    model.add(ZeroPadding2D((1,1)))                             #2
    model.add(Convolution2D(64, 3, 3, activation='relu'))       #3  -2
    model.add(MaxPooling2D((2,2), strides=(2,2)))               #4  /2

    model.add(ZeroPadding2D((1,1)))                             #5
    model.add(Convolution2D(128, 3, 3, activation='relu'))      #6  -1
    model.add(ZeroPadding2D((1,1)))                             #7
    model.add(Convolution2D(128, 3, 3, activation='relu'))      #8  -2
    model.add(MaxPooling2D((2,2), strides=(2,2)))               #9  /2

    model.add(ZeroPadding2D((1,1)))                             #10
    model.add(Convolution2D(256, 3, 3, activation='relu'))      #11
    model.add(ZeroPadding2D((1,1)))                             #12
    model.add(Convolution2D(256, 3, 3, activation='relu'))      #13
    model.add(ZeroPadding2D((1,1)))                             #14
    model.add(Convolution2D(256, 3, 3, activation='relu'))      #15
    model.add(MaxPooling2D((2,2), strides=(2,2)))               #16

    model.add(ZeroPadding2D((1,1)))                             #17
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #18
    model.add(ZeroPadding2D((1,1)))                             #19
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #20
    model.add(ZeroPadding2D((1,1)))                             #21
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #22
    model.add(MaxPooling2D((1,1), strides=(1,1)))               #23

    model.add(ZeroPadding2D((1,1)))                             #24
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #25
    model.add(ZeroPadding2D((1,1)))                             #26
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #27
    model.add(ZeroPadding2D((1,1)))                             #28
    model.add(Convolution2D(512, 3, 3, activation='relu'))      #29
    #model.add(MaxPooling2D((2,2), strides=(2,2)))               #30

    #model.add(Flatten())                                        #31
    #model.add(Dense(4096, activation='relu'))                   #32
    #model.add(Dropout(0.5))                                     #33
    #model.add(Dense(4096, activation='relu'))                   #34
    #model.add(Dropout(0.5))                                     #35
    #model.add(Dense(1000, activation='softmax'))                #36

    if weights_path:
        model.load_weights(weights_path)

    return model

def home_made_convnet(Xshape, Yshape):
    print 'initializing model...'

    model = Sequential()
    model.add(ZeroPadding2D((2, 2), input_shape=(Xshape)))  # 0
    model.add(Convolution2D(64, 5, 5, activation='relu'))  # 1
    model.add(Convolution2D(64, 5, 5, activation='relu'))  # 3
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))  # 4

    model.add(Convolution2D(128, 3, 3, activation='relu'))  # 6
    model.add(Convolution2D(128, 3, 3, activation='relu'))  # 8
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))  # 9

    model.add(Flatten())  # 31
    model.add(Dense(256, activation='relu'))  # 32
    model.add(Dropout(0.5))  # 33
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(Yshape, activation='softmax'))

    model.load_weights('weights_f.h5')

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def load_conv_model():
    conv_model = home_made_convnet((3, 50, 20), 26)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    conv_model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return conv_model


#https://groups.google.com/forum/#!msg/keras-users/7sw0kvhDqCw/QmDMX952tq8J
#A little modified so it can run on< 50 our examples
def pad_sequences(sequences, maxlen=None, dim=1, dtype='int32',
    padding='post', truncating='post', value=0.):
    '''
        Override keras method to allow multiple feature dimensions.

        @dim: input feature dimension (number of features per timestep)
    '''
    lengths = [s.shape[1] for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = np.zeros(shape=(nb_samples,maxlen,dim)).astype(dtype)
    #x = np.array(np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
    for i in range (0,nb_samples):
        for idx, s in enumerate(sequences[i]):
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % padding)

            if padding == 'post':
                x[i][:len(trunc)] = trunc
            elif padding == 'pre':
                x[i][idx, -len(trunc):] = trunc
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
    return x,maxlen


def recognize_word_file(wordfile, imgfile):
    n_words = matches = 0

    lines, _ = wordio.read(wordfile)
    img = Image.open(imgfile)

    output_lines = []

    for line_idx, line in enumerate(lines):
        output_line = []
        for word_idx, word in enumerate(line):
            output_word = copy.deepcopy(word)
            n_words += 1

            cropped = img.crop((word.left, word.top, word.right, word.bottom))
            color = np.copy(np.asarray(cropped))

            monograms, start, end = get_monograms_otsu(color, cropped)

            if monograms is None:
                continue

            for mg in monograms:
                width, height = 30, 55
                im = np.asarray(255 - mg.im, dtype=np.float32)
                im /= 255.

                im = scipy.misc.imresize(im, (height, im.shape[1], im.shape[2]))

                # if im.shape[1] <= width :
                # im = np.pad(im, ((0,0), (0,width-im.shape[1]), (0,0)), mode='constant', constant_values=0)
                # else :
                im = scipy.misc.imresize(im, (im.shape[0], width, im.shape[2]))

                # plt.imshow(im)
                # plt.show()

                im = np.asarray(im, dtype=np.float32)
                # im = im[:, :, :] - 126
                im /= 255.

                im = im.transpose((2, 0, 1))

                conv_output = conv_model.predict_classes(im.reshape((1, im.shape[0], im.shape[1], im.shape[2])),
                                                         batch_size=1, verbose=0)
                conv_confidences = conv_model.predict(im.reshape((1, im.shape[0], im.shape[1], im.shape[2])),
                                                      batch_size=1, verbose=0)
                letter = label_mapping[conv_output[0]]#chr(conv_output[0] + ord('a'))

                mg.set_prediction(letter)
                mg.set_confidence(conv_confidences[0][conv_output[0]])

                if len(monograms) < 50:  # So that it isn't that slow/freezes
                    threshold = 0.1

                    convident_idcs = conv_confidences > threshold

                    #charlist = (convident_idcs * range(len(label_mapping)))[convident_idcs] + ord('a')
                    char_idxs = (convident_idcs * range(len(label_mapping)))[convident_idcs]
                    chars = np.asarray(label_mapping)[char_idxs] #[convident_idcs] #label_mapping #[chr(char) for char in charlist]

                    confs = (conv_confidences)[convident_idcs]
                    options = zip(chars, confs)
                    mg.set_options(options)
                else:
                    mg.set_options([(letter, mg.get_confidence())])

            N_grams = monograms

            words = []

            word_prediction = ''
            #print 'Target: ', word.text,



            def build_words(wrd, N_grams, start, end):
                if wrd is None:
                    for g in N_grams:
                        if g.start == start:
                            build_words(g, N_grams, start, end)
                    return
                if wrd.end == end:
                    words.append(wrd)
                    return
                for idx, g in enumerate(N_grams):
                    if wrd.followed_by(g):
                        build_words(wrd.combine(g), N_grams[idx:], start, end)

            def build_words2(wrd, N_grams, start, end):
                if wrd is None:
                    for g in N_grams:
                        if g.start == start:
                            for char, conf in g.get_options():
                                cpy = N_gram(g.im, g.start, g.end, char, conf)
                                build_words2(cpy, N_grams, start, end)
                    return
                if wrd.end == end:
                    words.append(wrd)
                    return
                for idx, g in enumerate(N_grams):
                    if wrd.followed_by(g):
                        for char, conf in g.get_options():
                            cpy = N_gram(g.im, g.start, g.end, char, conf)
                            build_words2(wrd.combine(cpy), N_grams[idx:], start, end)

            build_words2(None, N_grams, start, end)

            words = sorted(words, key=lambda w: w.get_confidence())[::-1]

            # for w in words[:10]:
            #    print w.prediction, w.get_confidence()

            if len(words) == 0:
                print ''
                continue

            word_strings = [wrd.prediction for wrd in words]

            for wrd in word_strings:
                if wrd in vocabulary:
                    word_prediction = wrd
                    break

            if word_prediction == '':
                words = []

                def build_words(wrd, N_grams, start, end):
                    if wrd is None:
                        for g in N_grams:
                            if g.start == start:
                                build_words(g, N_grams, start, end)
                        return
                    if wrd.end == end:
                        words.append(wrd)
                        return
                    for idx, g in enumerate(N_grams):
                        if wrd.followed_by(g):
                            build_words(wrd.combine(g), N_grams[idx:], start, end)

                build_words(None, N_grams, start, end)

                words = sorted(words, key=lambda w: w.get_confidence())[::-1]

                word_strings = [wrd.prediction for wrd in words[:10]]

                word_exists = checkWordInNgrams2(word_strings, ngram_voc)

                if len(word_exists) == 1:
                    word_prediction = word_exists[0]
                else:
                    lev = calculateDistance(word_strings, vocabulary)
                    # print type(lev)
                    lev = sorted(lev.items(), key=operator.itemgetter(1))
                    counter = 0
                    # for k, v in lev[:10]:
                    #    print "word:" + k + "   with score:" + str(v)


                    if lev == {}:
                        if len(word_exists) >= 1:
                            word_prediction = word_exists[0]
                        else:
                            word_prediction = words[0].prediction
                    else:
                        closest = []
                        minDist = min([l[1] for l in lev])
                        for k, v in lev:
                            if v == minDist:
                                closest.append(k)
                        if len(closest) == 1:
                            word_prediction = closest[0]
                        else:
                            maxConf = 0
                            final_prediction = ''
                            for c in closest:
                                for w in words:
                                    if w.prediction == c and maxConf < w.get_confidence():
                                        maxConf = w.get_confidence()
                                        final_prediction = w.prediction

                            if final_prediction == '':
                                if len(word_exists) >= 1:
                                    word_prediction = word_exists[0]
                                else:
                                    word_prediction = words[0].prediction
                            else:
                                llev = calculateDistance([final_prediction], vocabulary)
                                llev = sorted(llev.items(), key=operator.itemgetter(1))
                                # print 'dictieeeee', llev[0][0]
                                word_prediction = llev[0][0]

            print 'Prediction: ', word_prediction
            output_word.text = word_prediction
            output_line.append(output_word)
        output_lines.append(output_line)

    return output_lines

if __name__ == "__main__":

    conv_model = load_conv_model()

    vocabulary = pickle.load(open('shen_vocabulary.pickle'))
    ngram_voc = extractNGrams(vocabulary)

    for f in listdir('subsettest/words'):
        wordfile = 'subsettest/words/' + f
        imgfile = wordfile.replace('.words', IM_EXT).replace('/words','/pages')

        output_lines = recognize_word_file(wordfile, imgfile)
        wordio.save(output_lines, 'outdir/' + f)





