from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD
#from keras.layers.extra import *
from keras.layers.recurrent import LSTM
import cPickle as pickle
import numpy as np
import random
from keras.utils.np_utils import to_categorical

#https://github.com/fchollet/keras/issues/401

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


trainX = dataX[:14000]
print trainX[0].shape
trainY = dataY[:14000]

testX = dataX[14000:]
testY = dataY[14000:]

model = Sequential()
#Input shape = (time_steps, n_dim)
left = Sequential()
left.add(LSTM(output_dim=512, return_sequences=True,
		input_shape=(None, 512)))
right = Sequential()
right.add(LSTM(output_dim=512, return_sequences=True, 
               input_shape=(None,512), go_backwards=True))

model.add(Merge([left, right], mode='concat'))

#model.add(LSTM(512, return_sequences=True,  input_shape=(1,512)))

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

for e in range(100):
  for i in range(len(trainX)):
    model.train_on_batch([trainX[i],trainX[i]], trainY[i:i+1])
  print 'epoch', e
  acc = 0
  for i in range(len(testX)):
    score, accuracy = model.test_on_batch([testX[i], testX[i]], testY[i:i+1])
    acc += accuracy
  print 'test accuracy: ', accuracy / float(len(testX))

