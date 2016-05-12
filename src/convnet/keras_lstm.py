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

dataX, dataY = pickle.load(open('c_features_keras.pkl'))
dataX = dataX.reshape((dataX.shape[0], 512))


data = zip(dataX, dataY)
random.shuffle(data)

dataX, dataY = zip(*data)
dataX = np.asarray(dataX)
rawY = dataY
dataY = to_categorical(np.asarray(dataY))

#Need input: (input_length,time_steps,n_dim)
dataX = np.reshape(dataX,(len(dataX),1,512))

print dataX.shape
print dataY.shape


trainX = dataX[:14000]
trainY = dataY[:14000]

testX = dataX[14000:]
testY = dataY[14000:]

model = Sequential()
#Input shape = (time_steps, n_dim)
model.add(LSTM(512, return_sequences=True, input_shape=(1,512)))

model.add(Dropout(0.2))

#Maybe thinking about return_sequences= True but we need then TimeDistributedDense
#Or we can do both LSTM next to each other and then merge them together.
model.add(LSTM(512, return_sequences=False))
model.add(Dense(dataY.shape[1]))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(trainX, trainY,nb_epoch=2500,batch_size=32)
score,accuracy = model.evaluate(testX, testY, batch_size=32)
print('The result of using evaluate method:', accuracy)
