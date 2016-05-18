from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
#from keras.layers.extra import *
from keras.layers.recurrent import LSTM
import cPickle as pickle
import numpy as np
import random
from keras.utils.np_utils import to_categorical



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
    dataX, dataY = pickle.load(open('c_features_keras.pkl'))
    #dataX = dataX.reshape((dataX.shape[0], 512))

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
                  optimizer=sgd,
                  metrics=['accuracy'])

    print "Train model"
    model.fit([trainX,trainX], trainY,batch_size=32,validation_split=0.1)
    print "Test model"
    score,accuracy = model.evaluate([testX,testX],testY,batch_size=32)
    print 'accuracy : ' + str(accuracy)