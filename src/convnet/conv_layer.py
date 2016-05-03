
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


class conv_layer(object):

    def __init__(self, rng, input, filter_shape, image_shape=None, \
                 pooling=False, poolsize=None,\
                 activation=T.nnet.relu, W=None, b=None, keepDims=False):
        if image_shape is not None:
            assert image_shape[1] == filter_shape[1] # Depth of image in batch has to match filter depth

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   (1 if not pooling else numpy.prod(poolsize)))
        # initialize weights with random weights
        if W is None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W = W

        if b is None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        if not keepDims:
            # convolve input feature maps with filters
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                input_shape=image_shape
            )
        else:
            tmp_conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                input_shape=image_shape,
                border_mode='full'
            )
            border = filter_shape[3] / 2
            if filter_shape[2] == 1:
                conv_out = tmp_conv_out[:, :, :, border:-border]
            else:
                border2 = filter_shape[2] / 2
                conv_out = tmp_conv_out[:, :, border2:-border2, border:-border]

        if pooling:
            assert(poolsize is not None)
            # downsample each feature map individually, using maxpooling
            pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=True
            )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation((conv_out if not pooling else pooled_out) + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def gated_loss(self, y):
        gates = T.nnet.sigmoid(self.output[:, 0:1, :, :])

        gated_square_loss   = T.mean(T.round(gates) * (self.output[:, 1:2, :, :] - y[:, 1:2, :, :]) ** 2)
        logistic_loss       = -T.mean(T.log(1 + T.exp(-y[:, 0:1, :, :] * gates)))
        return gated_square_loss, logistic_loss, gated_square_loss + logistic_loss

    def logistic_loss(self, y):
        return -T.mean(T.log(1 + T.exp(- y.flatten(2) * self.output.flatten(2))))

    def mse(self, y):
        return T.mean((self.output[:, :, :, 2:-2] - y[:, :, :, 2:-2]) ** 2)

    def gated_se(self, y):
        gates = T.largest(self.output[:, :, :, 2:-2], y[:, :, :, 2:-2])
        return T.sum(gates * (self.output[:, :, :, 2:-2] - y[:, :, :, 2:-2]) ** 2) / T.sum(gates)
