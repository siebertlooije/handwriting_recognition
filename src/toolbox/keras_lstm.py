from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
#from keras.layers.extra import *
from keras.layers.recurrent import LSTM
import cPickle as pickle
import numpy as np
import random
from keras.utils.np_utils import to_categorical
import gtk
import pamImage
import croplib
import wordio
import word as wrd
import os.path as pth
from scipy.ndimage.filters import gaussian_filter1d
from os import listdir
from matplotlib import pyplot

IM_EXT='.ppm'

import re
import numpy


from PIL import Image

import numpy as np


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

label_dict = {'!': 1, '#': 2, '$': 3, '&': 4, '*': 5, ',': 6, '-': 7, '.': 8, '4': 9, ':': 10, ';': 11, '?': 12,
                  '\\': 13, '^': 14, 'a': 15, 'b': 16, 'c': 17, 'd': 18, 'e': 19, 'f': 20, 'g': 21, 'h': 22, 'i': 23,
                  'k': 24, 'l': 25, 'm': 26, 'n': 27, 'o': 28, 'p': 29, 'q': 30, 'r': 31, 's': 32, 't': 33,
                  'u': 34, 'v': 35, 'w': 36, 'x': 37, 'y': 38, 'z': 39}


word_log = {}


def extractImages(wordfile, imgfile, l_file):
    print wordfile
    lines, _ = wordio.read(wordfile)
    img = Image.open(imgfile)
    # img = pamImage.PamImage(imgfile)

    # line_iter = iter(lines)
    # cur_line = line_iter.next()
    # word_iter = iter(cur_line)

    assert pth.exists('../crops/')
    out_str = '../crops/' + pth.basename(imgfile)
    out_str = out_str.replace(IM_EXT, '')
    train_y = []
    for line_idx, line in enumerate(lines):

        for word_idx, word in enumerate(line):
            word_array = np.zeros((word.right-word.left))

            for idx, char in enumerate(word.characters):

                word_array[char.left - word.left] = 1

            word_array[(word.right-word.left)-1] = 1

            gaussian_array = gaussian_filter1d(word_array,0.5)
            train_y.append(gaussian_array)

            """
                if 0 in cropped.size:
                    continue
                # print cropped.size
                otsu_img = img_as_ubyte(np.copy(np.asarray(cropped.convert('L'))))

                try:
                    threshold_global_otsu = threshold_otsu(otsu_img)
                except TypeError:
                    print 'Something weird happened'
                    continue
                global_otsu = otsu_img >= threshold_global_otsu

                l_otsu_im = 1 - global_otsu

                index_row = np.tile(np.arange(otsu_img.shape[0]).reshape(otsu_img.shape[0], 1), (1, otsu_img.shape[1]))
                index_col = np.tile(np.arange(otsu_img.shape[1]), (otsu_img.shape[0], 1))

                non0 = l_otsu_im.nonzero()
                Wrow = np.multiply(l_otsu_im[non0], index_row[non0])
                Wcol = np.multiply(l_otsu_im[non0], index_col[non0])

                Mrow = np.mean(Wrow)
                Mcol = np.mean(Wcol)

                Std_row = np.std(Wrow)
                Std_col = np.std(Wcol)

                top = max(0, int(Mrow - 3 * Std_row))
                bottom = min(cropped.size[1], int(Mrow + 3 * Std_row))

                left = max(0, int(Mcol - 3 * Std_col))
                right = min(cropped.size[0], int(Mcol + 3 * Std_col))

                sub_cropped = cropped.crop((left, top, right, bottom))

                out_path = out_str + '_l' + str(line_idx) + '_w' + str(word_idx) + '_c' + str(
                    idx) + '_t_' + char.text + IM_EXT
                try:
                    sub_cropped.save(out_path)
                except SystemError:
                    print 'bad annotation...'

                if char.text in word_log:
                    word_log[char.text.lower()] += 1
                else:
                    word_log[char.text.lower()] = 1

                if char.text.lower() in label_dict:
                    l_file.wri te(out_str + ' ' + str(label_dict[char.text.lower()] - 1) + '\n')
                    """
    return train_y

#https://github.com/fchollet/keras/issues/401
if __name__ == "__main__":

    label_file = open('labels.txt', 'w')
    label_file.write('# label file\n')

    print 'Cropping...'
    assert pth.exists('../charannotations/KNMP') and pth.exists('../charannotations/Stanford')


    for f in listdir('../charannotations/KNMP'):
        wordfile = '../charannotations/KNMP/' + f
        imgfile = wordfile.replace('.words', IM_EXT).replace('/charannotations', '/pages').replace('2C20', '2C2O')

        train_y = extractImages(wordfile, imgfile, label_file)

    for f in listdir('../charannotations/Stanford'):
        wordfile = '../charannotations/Stanford/' + f
        imgfile = wordfile.replace('.words', IM_EXT).replace('/charannotations', '/pages').replace('2C20', '2C2O')

        train_y =  extractImages(wordfile, imgfile, label_file)

    feature_data = {}

    features = []
    labels = []
    for i, line in enumerate(open('../toolbox/labels.txt')):
        path, label = line.split()
        try:
            im = np.asarray(Image.open(path.replace('/crops', '/crops/letters')), dtype=np.float32)
            if im.shape[0] < 10 or im.shape[1] < 10:
                continue
        except IOError:
            print(path + ' gives trouble...')
            continue

        im = im[:, :, ::-1]
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68

        im = im.transpose((2, 0, 1))

        # run lasagne
        print "Computing keras result... Im shape:", im.shape,

        keras_result = model.predict(im.reshape((1, im.shape[0], im.shape[1], im.shape[2])))
        print 'result shape', keras_result.shape
        features.append(np.mean(keras_result, axis=(2)).transpose((0, 2, 1)))
        labels.append(int(label))





    print 'done cropping'
    print 'overview of words: '

    """
    print dataX[0].shape


    data = zip(dataX, dataY)
    random.shuffle(data)

    dataX, dataY = zip(*data)
    #dataX = np.asarray(dataX)
    rawY = dataY
    dataY = to_categorical(np.asarray(dataY))

    #Need input: (input_length,time_steps,n_dim)
    #dataX = np.reshape(dataX,(len(dataX),1,512))

    #print dataX.shape
    #print dataY.shape
    dataX,maxlen = pad_sequences(dataX,dim=512)

    trainX = dataX[:14000]
    print trainX[0].shape
    trainY = dataY[:14000]

    testX = dataX[14000:]
    testY = dataY[14000:]

    print "Building up model"

    model = Sequential()
    #Input shape = (time_steps, n_dim)
    left = Sequential()
    left.add(LSTM(output_dim=512, return_sequences=True,
            input_shape=(maxlen, 512)))
    right = Sequential()
    right.add(LSTM(output_dim=512, return_sequences=True,
                   input_shape=(maxlen,512), go_backwards=True))

    model.add(Merge([left, right], mode='concat'))

    #model.add(LSTM(512, return_sequences=True,  input_shape=(1,512)))

    model.add(Dropout(0.2))

    #Maybe thinking about return_sequences= True but we need then TimeDistributedDense
    #Or we can do both LSTM next to each other and then merge them together.
    model.add(LSTM(512, return_sequences=False))
    model.add(Dense(dataY.shape[1]))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    print "Compiled model"
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print "Train model"
    model.fit([trainX,trainX], trainY,batch_size=32,validation_split=0.1, nb_epoch=1000)
    print "Test model"
    score,accuracy = model.evaluate([testX,testX],testY,batch_size=32)
    print 'accuracy : ' + str(accuracy)  """
