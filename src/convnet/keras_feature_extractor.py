from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from PIL import Image

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))   #0
    model.add(Convolution2D(64, 3, 3, activation='relu'))       #1
    model.add(ZeroPadding2D((1,1)))                             #2
    model.add(Convolution2D(64, 3, 3, activation='relu'))       #3
    model.add(MaxPooling2D((2,2), strides=(2,2)))               #4

    model.add(ZeroPadding2D((1,1)))                             #5
    model.add(Convolution2D(128, 3, 3, activation='relu'))      #6
    model.add(ZeroPadding2D((1,1)))                             #7
    model.add(Convolution2D(128, 3, 3, activation='relu'))      #8
    model.add(MaxPooling2D((2,2), strides=(2,2)))               #9

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

if __name__ == "__main__":
    import cPickle as pickle
    import h5py
    # Test pretrained model
    f = h5py.File('vgg16_weights.h5', 'r+')
    f.attrs['nb_layers'] = 30
    f.close()

    model = VGG_16('vgg16_weights.h5')

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

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
        features.append(np.mean(keras_result, axis=2).transpose((0, 2, 1)))
        #print features[0].shape
        labels.append(int(label))

    feature_file = open('c_features_keras.pkl', 'wb')
    pickle.dump((features, np.asarray(labels, dtype='int64')), feature_file)
    feature_file.close()

    #out = model.predict(im)
    #print np.argmax(out)