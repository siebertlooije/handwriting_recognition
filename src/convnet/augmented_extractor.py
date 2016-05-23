from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from PIL import Image
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras.layers.core import Merge

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


def augment(myim):
    meanInk = np.zeros(3)
    meanPaper = np.ones(3) * 255
    def assign_points():
        distInk = myim - meanInk
        distPap = myim - meanPaper

        distInk **= 2
        distPap **= 2

        return np.sum(distInk, axis=2) > np.sum(distPap, axis=2)

    points = assign_points()
    mask = np.tile(points, (3, 1, 1)).transpose((1, 2, 0))
    augmented = np.copy(myim)
    if np.random.rand() > .5:
        augmented[mask] *= (.75 + np.random.rand() * .5)
        augmented[mask == 0] *= np.random.rand() * .5 + .75
    else:
        augmented[mask] += (np.random.randint(20) - 40)
        augmented[mask == 0] += (np.random.randint(20) - 40)

    augmented = np.minimum(augmented, 255)
    augmented = np.maximum(augmented, 0)
    return augmented

def get_augmented_set():
    print 'augmenting...',


    feature_data = {}

    features = []
    labels = []
    images = []

    for i, line in enumerate(open('../toolbox/labels_shuf.txt')):
        if i > 12000:
            break
        path, label = line.split()
        try:
            im = np.asarray(Image.open(path.replace('/crops', '/crops/letters')), dtype=np.float32)
            if im.shape[0] < 10 or im.shape[1] < 10:
                continue
        except IOError:
            print(path + ' gives trouble...')
            continue

        im = augment(im)
        im = im[:, :, ::-1]
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68

        im = im.transpose((2, 0, 1))

        # run lasagne
        #print "Computing keras result... Im shape:", im.shape,
        #plt.imshow(im.transpose((1, 2, 0)))
        #splt.show()
        keras_result = conv_model.predict(im.reshape((1, im.shape[0], im.shape[1], im.shape[2])))
        #print 'result shape', keras_result.shape

        if keras_result.shape[2] < 2:
            continue

        features.append(
            np.concatenate((np.max(keras_result[:, :, :keras_result.shape[2] / 2, :], axis=2),
                            np.max(keras_result[:, :, keras_result.shape[2] / 2:, :], axis=2)), axis=1).transpose((0, 2, 1)))
        labels.append(int(label))

    print 'done!'

    return features, np.asarray(labels, dtype='int64')



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


#https://github.com/fchollet/keras/issues/401
if __name__ == "__main__":

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
    lstm_model.add(Dense(39))
    lstm_model.add(Activation('softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    print "Compiled model"
    lstm_model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    for epoch in range(100):
        dataX, dataY = get_augmented_set()

        dataY = to_categorical(np.asarray(dataY))

        dataX, _ = pad_sequences(dataX,dim=1024, maxlen=maxlen)

        print "Train model, global epoch: ", epoch
        lstm_model.fit([dataX,dataX], dataY,batch_size=32,validation_split=0.1, nb_epoch=1)

    #print "Test model"
    #score,accuracy = model.evaluate([testX,testX],testY,batch_size=32)
    #print 'accuracy : ' + str(accuracy)