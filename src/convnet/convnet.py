"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

from __future__ import print_function

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

from conv_layer import conv_layer

import six.moves.cPickle as pickle
import matplotlib.pyplot as plt

from spatial_pyramid_pooling import SPP
from PIL import Image

from random import shuffle

def load_char_data():
    imgs = []
    labels = []
    for line in open('../toolbox/labels.txt'):
        path, label = line.split()
        imgs.append(numpy.asarray(Image.open(path)))
        labels.append(int(label))

    data = shuffle(zip(imgs, labels))

    imgs, labels = zip(*data)

    train_x = imgs[0:int(.6 * len(imgs))]
    train_y = labels[0:int(.6 * len(imgs))]

    test_x = imgs[int(.6 * len(imgs)):int(.8 * len(imgs))]
    test_y = labels[int(.6 * len(imgs)):int(.8 * len(imgs))]

    val_x = imgs[int(.6 * len(imgs)):int(.8 * len(imgs))]
    val_y = labels[int(.6 * len(imgs)):int(.8 * len(imgs))]

    def shared_dataset(data, labels, borrow=True):

        shared_data = [theano.shared(img.reshape(1, 1 if len(img.shape) == 2 else 3, img.shape[-2], img.shape[-1]),
                                     dtype=theano.config.floatX)
                       for img in data]
        shared_labels = theano.shared(numpy.asarray(labels,
                                                    dtype=theano.config.floatX))
        return shared_data, T.cast(shared_labels, 'int32')

    train_x_r, train_y_r = shared_dataset(train_x, train_y)
    test_x_r, test_y_r = shared_dataset(test_x, test_y)
    val_x_r, val_y_r = shared_dataset(val_x, val_y)

    return train_x_r, train_y_r, test_x_r, test_y_r, val_x_r, val_y_r


def load_latline_dataset():
    f = open('data.pickle', 'rb')

    train_data, train_labels, test_data, test_labels, val_data, val_labels = pickle.load(f)

    def shared_dataset(data, labels, borrow=True):
        shared_data     = theano.shared(numpy.asarray(data,
                                                      dtype=theano.config.floatX))
        shared_labels   = theano.shared(numpy.asarray(labels,
                                                      dtype=theano.config.floatX))
        return shared_data, shared_labels

    train_x, train_y = shared_dataset(train_data, train_labels)
    test_x, test_y  = shared_dataset(test_data, test_labels)
    val_x, val_y = shared_dataset(val_data, val_labels)

    rval = [(train_x, train_y), (test_x, test_y), (val_x, val_y)]
    return rval, len(train_data.shape) - 1, [1, train_data.shape[1]]



def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1

    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + T.grad(cost, param)))
        #updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates



def evaluate_convnet(learning_rate=0.02, n_epochs=2000,
                    dataset='single_sphere',
                    nkerns=[32, 64, 64, 128], batch_size=500,
                    filter_shapes=[[5, 5], [5, 5], [3, 3], [3, 3]], momentum=0.9, half_time=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets, depth_dim, conv_dims = load_latline_dataset() # << TODO implement

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as spiking of sensors at lateral line
    y = T.matrix('y')   # The output is the distance (in x- and y-directions) of sphere

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of sensor detections to a 4D tensor
    layer0_input = x.reshape((batch_size, depth_dim, conv_dims[0], conv_dims[1]))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    print (nkerns[0], depth_dim, filter_shapes[0][0], filter_shapes[0][1])

    print (batch_size, depth_dim, conv_dims[0], conv_dims[1])
    print (nkerns[0], depth_dim, filter_shapes[0][0], filter_shapes[0][1])

    layer0 = conv_layer(
        rng,
        input=layer0_input,
        image_shape=None,
        filter_shape=(nkerns[0], 3, filter_shapes[0][0], filter_shapes[0][1]),
        pooling=False,
        activation=T.nnet.relu
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)

    layer1 = conv_layer(
        rng,
        input=layer0.output,
        image_shape=None,
        filter_shape=(nkerns[1], nkerns[0], filter_shapes[1][0], filter_shapes[1][1]),
        pooling=True,
        poolsize=(2, 2),
        activation=T.nnet.relu,
        keepDims=True
    )

    layer1b = conv_layer(
        rng,
        input=layer1.output,
        image_shape=None,
        filter_shape=(nkerns[2], nkerns[1], filter_shapes[2][0], filter_shapes[2][1]),
        pooling=False,
        activation=T.nnet.relu,
        keepDims=True
    )

    layer1c = conv_layer(
        rng,
        input=layer1b.output,
        image_shape=None,
        filter_shape=(nkerns[3], nkerns[2], filter_shapes[3][0], filter_shapes[3][1]),
        pooling=False,
        activation=T.nnet.relu,
        keepDims=True
    )

    spp_layer = SPP(layer1c.output)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = spp_layer.output

    # construct a fully-connected ReLU layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=spp_layer.M * nkerns[-1],
        n_out=500,
        activation=T.nnet.relu
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=39)

    # linear regression by using a fully connected layer
    layer3 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=conv_dims[1] * 2,
        n_out=2,
        activation=None
    )

    # classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.mean_squared_error(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        cost,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    demo_model = theano.function(
        [index],
        [layer3.output, y],
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params + layer1b.params + layer1c.params

    # create a list of gradients for all model parameters
    #grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    #updates = [
    #    (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]

    l_r = T.scalar('l_r', dtype=theano.config.floatX)

    updates = gradient_updates_momentum(cost, params, l_r, momentum)


    train_model = theano.function(
        [index, l_r],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        if epoch % half_time == 0:
            learning_rate /= 2

        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index, learning_rate)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, MSE %f ' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test MSE of '
                           'best model %f ') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation MSE of %f %% obtained at iteration %i, '
          'with test MSE %f ' %
          (best_validation_loss, best_iter + 1, test_score))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    demo_outputs = [
        demo_model(i)
        for i in range(n_test_batches)
    ]

    sensor_range = [-1.5, 1.5]
    y_range = [0, 1]
    plt.ion()

    plotting = False

    MED = 0
    for i in range(n_test_batches):
        predicted, target = demo_outputs[i]
        for j in range(predicted.shape[0]):
            x_hat, y_hat = predicted[j]
            x, y = target[j]

            MED += numpy.sqrt((x - x_hat) ** 2 + (y - y_hat) ** 2)

            if plotting:
                plt.clf()
                plt.plot([x_hat], [y_hat], 'ro')
                plt.plot([x], [y], 'g+')
                plt.grid()
                plt.axis([sensor_range[0], sensor_range[1], y_range[0], y_range[1]])
                plt.pause(0.05)
    MED /= 2000

    print('MED = %f\n' % MED)


if __name__ == '__main__':
    evaluate_convnet()
