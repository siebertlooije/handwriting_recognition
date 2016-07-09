from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta
import cv2, numpy as np, scipy
from PIL import Image

from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

import os
import os.path
from keras.preprocessing.image import ImageDataGenerator

import keras_lstm

def visualize_data_sizes():
    ws = np.asarray([])
    hs = np.asarray([])
    maxw = 0
    maxh = 0

    for i, line in enumerate(open('../toolbox/labels.txt')):
        path, label = line.split()
        try:
            im = np.asarray(Image.open(path.replace('/crops', '/crops/letters'))) # , dtype=np.float32

            ws = np.append(ws, im.shape[0])
            hs = np.append(hs, im.shape[1])

            if im.shape[0] > maxw :
                maxw = im.shape[0]
            if im.shape[1] > maxh :
                maxh = im.shape[1]

            # if im.shape[0] > 150 or im.shape[1] >150:
            #     plt.imshow(im)
            #     plt.show()
            #     print label
            #     continue
        except IOError:
            print(path + ' gives trouble...')
            continue

    plt.hist(ws, bins = 40)
    plt.show()
    print (maxw, maxh)
    plt.hist(hs, bins = 40)
    plt.show()
    
def init_model(Xshape, Yshape) :
    print 'initializing model...'

    model = Sequential()
    model.add(ZeroPadding2D((2,2),input_shape=(Xshape)))   #0
    model.add(Convolution2D(64, 5, 5, activation='relu'))       #1
    model.add(Convolution2D(64, 5, 5, activation='relu'))       #3
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))               #4

    model.add(Convolution2D(128, 3, 3, activation='relu'))      #6
    model.add(Convolution2D(128, 3, 3, activation='relu'))      #8
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))               #9

    model.add(Flatten())                                        #31
    model.add(Dense(256, activation='relu'))                    #32
    model.add(Dropout(0.5))                                     #33
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(Yshape, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #adadelta = Adadelta()
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# pads if im.shape[1||2] < width || height, resizes otherwise
def resize_dataset(width, height) :
    counter = 0

    print 'counting images...'
    heights = []
    width = []
    for i, line in enumerate(open('../toolbox/labels_shuf_new.txt')):
        path, label = line.split()
        try:
            im = np.asarray(Image.open(path.replace('/crops', '/crops/letters'))) # , dtype=np.float32
            heights.append(im.shape[0])
            width.append(im.shape[1])
            #print im.shape
            counter += 1
        except IOError:
            print(path + ' gives trouble...')
            continue

    height = int(np.median(heights))
    width = int(np.median(width))
    print 'w x h == {0} x {1}'.format(width, height)


    X = np.zeros((counter, 3, height, width))
    Y = np.zeros((counter, 3, height, width), dtype = int)
    
    total = counter
    print str(total) + ' images found'

    counter = 0
    for i, line in enumerate(open('../toolbox/labels_shuf_new.txt')):
        path, label = line.split()
        try:

            im = np.asarray(Image.open(path.replace('/crops', '/crops/letters')), dtype=np.float32) # , dtype=np.float32

            #print np.max(im), np.min(im),
            im = 255 - im
            #im = im[:, :, :] - 126
            im /= 255.

            im = scipy.misc.imresize(im, (height, im.shape[1], im.shape[2]))

            im = scipy.misc.imresize(im, (im.shape[0], width, im.shape[2]))

            im = np.asarray(im, dtype=np.float32)
            # im = im[:, :, :] - 126
            im /= 255.

            im = im.transpose((2, 0, 1))
            
            X[counter] = im
            Y[counter] = label



            counter += 1

            progress = counter/float(total)
            loadbar = '#' * int(round(20*progress)) +  ' ' * int(round(20*(1-progress)))
            print '\r[{0}] {1} of {2} images resized'.format(loadbar,
                counter,
                total),
        except IOError:
            continue
    print 
    return X,Y

if __name__ == "__main__":
    width, height = 20, 50

    X,Y = resize_dataset(width, height)
    Y = to_categorical(np.asarray(Y))

    trainX = X#[:14000]
    trainY = Y#[:14000]

    #testX = X[14000:]
    #testY = Y[14000:]

    model = init_model(X.shape[1:], Y.shape[1])

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=2,
        shear_range=.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        ink_intensity=True,
        fill_mode='constant'
    )

    datagen.fit(trainX)

    model.fit_generator(datagen.flow(trainX, trainY, batch_size=32), samples_per_epoch=len(trainX), nb_epoch=300)

    #model.fit(X, Y, batch_size = 16, nb_epoch = 100, validation_split= .1)

    filedir = os.path.join(os.getcwd())
    filename = os.path.join(filedir, 'weights_f.h5')
    model.save_weights(filename)
