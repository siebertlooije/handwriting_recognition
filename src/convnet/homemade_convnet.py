from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np, scipy
from PIL import Image

from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

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
    model.add(ZeroPadding2D((1,1),input_shape=(Xshape)))   #0
    model.add(Convolution2D(64, 3, 3, activation='relu'))       #1
    model.add(ZeroPadding2D((1,1)))                             #2
    model.add(Convolution2D(64, 3, 3, activation='relu'))       #3
    model.add(MaxPooling2D((2,2), strides=(1,1)))               #4

    model.add(ZeroPadding2D((1,1)))                             #5
    model.add(Convolution2D(128, 3, 3, activation='relu'))      #6
    model.add(ZeroPadding2D((1,1)))                             #7
    model.add(Convolution2D(128, 3, 3, activation='relu'))      #8
    model.add(MaxPooling2D((2,2), strides=(1,1)))               #9

    model.add(ZeroPadding2D((1,1)))                             #10
    model.add(Convolution2D(256, 3, 3, activation='relu'))      #11
    model.add(ZeroPadding2D((1,1)))                             #12
    model.add(Convolution2D(256, 3, 3, activation='relu'))      #13
    model.add(ZeroPadding2D((1,1)))                             #14
    model.add(Convolution2D(256, 3, 3, activation='relu'))      #15
    model.add(MaxPooling2D((2,2), strides=(1,1)))               #16

    # model.add(ZeroPadding2D((1,1)))                             #17
    # model.add(Convolution2D(512, 3, 3, activation='relu'))      #18
    # model.add(ZeroPadding2D((1,1)))                             #19
    # model.add(Convolution2D(512, 3, 3, activation='relu'))      #20
    # model.add(ZeroPadding2D((1,1)))                             #21
    # model.add(Convolution2D(512, 3, 3, activation='relu'))      #22
    # model.add(MaxPooling2D((1,1), strides=(1,1)))               #23

    # model.add(ZeroPadding2D((1,1)))                             #24
    # model.add(Convolution2D(512, 3, 3, activation='relu'))      #25
    # model.add(ZeroPadding2D((1,1)))                             #26
    # model.add(Convolution2D(512, 3, 3, activation='relu'))      #27
    # model.add(ZeroPadding2D((1,1)))                             #28
    # model.add(Convolution2D(512, 3, 3, activation='relu'))      #29
    # model.add(MaxPooling2D((2,2), strides=(2,2)))               #30

    model.add(Flatten())                                        #31
    model.add(Dense(128, activation='relu'))                    #32
    model.add(Dropout(0.5))                                     #33

    model.add(Dense(Yshape, activation='softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, 
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

# pads if im.shape[1||2] < width || height, resizes otherwise
def resize_dataset(width, height) :
    counter = 0

    print 'counting images...'
    for i, line in enumerate(open('../toolbox/labels.txt')):
        path, label = line.split()
        try:
            im = np.asarray(Image.open(path.replace('/crops', '/crops/letters'))) # , dtype=np.float32
            counter += 1
        except IOError:
            print(path + ' gives trouble...')
            continue

    X = np.zeros((counter, 3, width, height))
    Y = np.zeros((counter, 3, width, height), dtype = int)
    
    total = counter
    print str(total) + ' images found'

    counter = 0
    for i, line in enumerate(open('../toolbox/labels.txt')):
        path, label = line.split()
        try:

            im = np.asarray(Image.open(path.replace('/crops', '/crops/letters')), dtype=np.float32) # 
            
            im = 255 - im[:, :, :]

            if im.shape[0] <= width :
                im = np.pad(im, ((0,width-im.shape[0]), (0,0), (0,0)), mode='constant', constant_values=0)
            else :
                im = scipy.misc.imresize(im, (width, im.shape[1], im.shape[2]))

            if im.shape[1] <= height :
                im = np.pad(im, ((0,0), (0,height-im.shape[1]), (0,0)), mode='constant', constant_values=0)
            else :
                im = scipy.misc.imresize(im, (im.shape[0], height, im.shape[2]))

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
    width, height = 32, 32

    X,Y = resize_dataset(width, height)
    Y = to_categorical(np.asarray(Y))
    
    model = init_model(X.shape[1:], Y.shape[1])


    model.fit(X, Y, batch_size = 32, nb_epoch = 10, validation_split= .1)
    
    