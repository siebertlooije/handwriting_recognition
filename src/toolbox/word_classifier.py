from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from PIL import Image
from keras.layers import *
from keras.layers.core import Merge
import os.path as pth
from skimage.util import img_as_ubyte
import wordio
from skimage.filters import threshold_otsu

import sys
from scipy.signal import argrelextrema


class N_gram(object):

    def __init__(self, im, start, end, prediction=None):
        self.im = im
        self.start = start
        self.end = end

        self.prediction = prediction

    def set_prediction(self, pred):
        self.prediction = pred

    def combine(self, other):
        return N_gram(self.im, self.start, other.end, self.prediction + other.prediction)

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
        continue
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

    monograms = []

    cuts = maxes[0]
    for idx, cut in enumerate(cuts):
        for next in cuts[idx + 1:]:
            print next, cut
            if 10 < next - cut:
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


import matplotlib.pyplot as plt

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


import cPickle as pickle
import h5py
# Test pretrained model
f = h5py.File('vgg16_weights.h5', 'r+')
f.attrs['nb_layers'] = 30
f.close()

conv_model = VGG_16('vgg16_weights.h5')

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
conv_model.compile(optimizer=sgd, loss='categorical_crossentropy')

#https://groups.google.com/forum/#!msg/keras-users/7sw0kvhDqCw/QmDMX952tq8J
#A little modified so it can run on our examples
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


if __name__ == "__main__":
    wordfile = sys.argv[1]
    imgfile = sys.argv[2]

    assert pth.exists(wordfile) and pth.exists(imgfile)

    lstm_model = Sequential()
    # Input shape = (time_steps, n_dim)
    maxlen=31
    left = Sequential()
    left.add(LSTM(output_dim=512, return_sequences=True,
                  input_shape=(maxlen, 1024)))
    right = Sequential()
    right.add(LSTM(output_dim=512, return_sequences=True,
                   input_shape=(maxlen, 1024), go_backwards=True))

    lstm_model.add(Merge([left, right], mode='concat'))

    # model.add(LSTM(512, return_sequences=True,  input_shape=(1,512)))

    lstm_model.add(Dropout(0.2))

    # Maybe thinking about return_sequences= True but we need then TimeDistributedDense
    # Or we can do both LSTM next to each other and then merge them together.
    lstm_model.add(LSTM(512, return_sequences=False))
    lstm_model.add(Dense(26))
    lstm_model.add(Activation('softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    print 'loading weights'
    lstm_model.load_weights('lstm_weights.h5')

    print "Compiled model"
    lstm_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',
                       metrics=['accuracy'])

    lines, _ = wordio.read(wordfile)
    img = Image.open(imgfile)

    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line):
            cropped = img.crop((word.left, word.top, word.right, word.bottom))
            color = np.copy(np.asarray(cropped))

            monograms, start, end = get_monograms_otsu(color, cropped)

            for mg in monograms:
                im = 255 - mg.im[:, :, ::-1]

                im[:, :, 0] -= 103.939
                im[:, :, 1] -= 116.779
                im[:, :, 2] -= 123.68

                im = im.transpose((2, 0, 1))
                convnet_output = conv_model.predict(im.reshape((1, im.shape[0], im.shape[1], im.shape[2])))

                features = np.concatenate((np.max(convnet_output[:, :, :convnet_output.shape[2] / 2, :], axis=2),
                                           np.max(convnet_output[:, :, convnet_output.shape[2] / 2:, :], axis=2)), axis=1).transpose((0, 2, 1))

                feature_seq, _ = pad_sequences([features], maxlen=31, dim=1024)

                lstm_output = lstm_model.predict_classes([feature_seq, feature_seq], batch_size=1, verbose=0)

                letter = chr(lstm_output[0] + ord('a'))

                #print letter
                mg.set_prediction(letter)

                #print mg.prediction

            words = []

            def build_words(wrd, monograms, start, end):
                if wrd is None:
                    for g in monograms:
                        if g.start == start:
                            build_words(g, monograms, start, end)
                    return
                if wrd.end == end:
                    words.append(wrd)
                    return
                for g in monograms:
                    if wrd.followed_by(g):
                        build_words(wrd.combine(g), monograms, start, end)

            build_words(None, N_grams, start, end)

            print '... done'

            for word in words:
                print word.prediction

            plt.imshow(color)
            plt.show()







