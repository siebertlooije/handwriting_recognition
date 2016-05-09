from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
from keras.layers.extra import *
from keras.layers.recurrent import LSTM
import cPickle as pickle
import numpy as np

import random
from keras.utils.np_utils import to_categorical

#https://github.com/fchollet/keras/issues/401

dataX, dataY = pickle.load(open('c_features_keras_timedistributed.pkl'))
dataX = dataX.reshape((dataX.shape[0], 512))


data = zip(dataX, dataY)
random.shuffle(data)

dataX, dataY = zip(*data)
dataX = np.asarray(dataX)
rawY = dataY
dataY = to_categorical(np.asarray(dataY))

print dataX.shape
print dataY.shape

trainX = dataX[:14000]
trainY = dataY[:14000]

testX = dataX[14000:]
testY = dataY[14000:]

model = Sequential()
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=True))
model.add(TimeDistributedDense(600))
model.add(Activation('softmax'))

#model = Sequential([
#    Dense(64, input_dim=512),
#    Activation('relu'),
#    Dense(np.max(rawY) + 1),
#    Activation('softmax')
#])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(trainX, trainY, nb_epoch=2500, batch_size=32)
score,accuracy = model.evaluate(testX, testY, batch_size=32)
print('The result of using evaluate method:', accuracy)
