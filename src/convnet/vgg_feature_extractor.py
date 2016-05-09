

from lasagne.layers import InputLayer, Conv2DLayer
from lasagne.layers import MaxPool2DLayer, LocalResponseNormalization2DLayer
from lasagne.layers import SliceLayer, concat, DenseLayer
import lasagne.nonlinearities
from os import listdir

from PIL import Image

def build_model():
    net = {}
    net['data'] = InputLayer(shape=(None, 3, None, None))

    # conv1
    net['conv1'] = Conv2DLayer(
        net['data'],
        num_filters=96,
        filter_size=(11, 11),
        stride = 4,
        nonlinearity=lasagne.nonlinearities.rectify)


    # pool1
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=(3, 3), stride=2)

    # norm1
    net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'],
                                                     n=5,
                                                     alpha=0.0001/5.0,
                                                     beta = 0.75,
                                                     k=1)

    # conv2
    # The caffe reference model uses a parameter called group.
    # This parameter splits input to the convolutional layer.
    # The first half of the filters operate on the first half
    # of the input from the previous layer. Similarly, the
    # second half operate on the second half of the input.
    #
    # Lasagne does not have this group parameter, but we can
    # do it ourselves.
    #
    # see https://github.com/BVLC/caffe/issues/778
    # also see https://code.google.com/p/cuda-convnet/wiki/LayerParams

    # before conv2 split the data
    net['conv2_data1'] = SliceLayer(net['norm1'], indices=slice(0, 48), axis=1)
    net['conv2_data2'] = SliceLayer(net['norm1'], indices=slice(48,96), axis=1)

    # now do the convolutions
    net['conv2_part1'] = Conv2DLayer(net['conv2_data1'],
                                     num_filters=128,
                                     filter_size=(5, 5),
                                     pad = 2)
    net['conv2_part2'] = Conv2DLayer(net['conv2_data2'],
                                     num_filters=128,
                                     filter_size=(5,5),
                                     pad = 2)

    # now combine
    net['conv2'] = concat((net['conv2_part1'],net['conv2_part2']),axis=1)

    # pool2
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=(3, 3), stride = 2)

    # norm2
    net['norm2'] = LocalResponseNormalization2DLayer(net['pool2'],
                                                     n=5,
                                                     alpha=0.0001/5.0,
                                                     beta = 0.75,
                                                     k=1)

    # conv3
    # no group
    net['conv3'] = Conv2DLayer(net['norm2'],
                               num_filters=384,
                               filter_size=(3, 3),
                               pad = 1)

    # conv4
    # group = 2
    net['conv4_data1'] = SliceLayer(net['conv3'], indices=slice(0, 192), axis=1)
    net['conv4_data2'] = SliceLayer(net['conv3'], indices=slice(192,384), axis=1)
    net['conv4_part1'] = Conv2DLayer(net['conv4_data1'],
                                     num_filters=192,
                                     filter_size=(3, 3),
                                     pad = 1)
    net['conv4_part2'] = Conv2DLayer(net['conv4_data2'],
                                     num_filters=192,
                                     filter_size=(3,3),
                                     pad = 1)
    net['conv4'] = concat((net['conv4_part1'],net['conv4_part2']),axis=1)

    # conv5
    # group 2
    net['conv5_data1'] = SliceLayer(net['conv4'], indices=slice(0, 192), axis=1)
    net['conv5_data2'] = SliceLayer(net['conv4'], indices=slice(192,384), axis=1)
    net['conv5_part1'] = Conv2DLayer(net['conv5_data1'],
                                     num_filters=128,
                                     filter_size=(3, 3),
                                     pad = 1)
    net['conv5_part2'] = Conv2DLayer(net['conv5_data2'],
                                     num_filters=128,
                                     filter_size=(3,3),
                                     pad = 1)
    net['conv5'] = concat((net['conv5_part1'],net['conv5_part2']),axis=1)

    # pool 5
    net['pool5'] = MaxPool2DLayer(net['conv5'], pool_size=(3, 3), stride = 2)

    return net

def load_caffe_weights():
    """
        Loads the parameters for the caffe reference model.
        Also checks to make sure the two models produce equivalent
        results. The ouput model is saved to caffe_reference.pkl

        Before running this, you need to install caffe and change
        the caffe_root variable to point to your installation.

        http://caffe.berkeleyvision.org/
        """

    import cPickle as pickle
    import lasagne
    import numpy as np
    from lasagne.utils import floatX

    # Make sure that caffe is on the python path:
    caffe_root = '/home/jos/caffe/'
    import sys
    sys.path.insert(0, caffe_root + 'python')
    import caffe

    # load caffenet
    caffe.set_mode_cpu()
    caffe_net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                          caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                          caffe.TEST)

    # create lasagne network
    my_net = build_model()

    print 'Loading pretrained weights'
    # try it the way they do in the recipies
    layers_caffe = dict(zip(list(caffe_net._layer_names), caffe_net.layers))
    for name, layer in my_net.items():
        try:
            if name == 'conv2':
                W = layers_caffe[name].blobs[0].data[:, :, ::-1, ::-1]
                b = layers_caffe[name].blobs[1].data

                my_net['conv2_part1'].W.set_value(W[0:128, :, :, :])
                my_net['conv2_part1'].b.set_value(b[0:128])
                my_net['conv2_part2'].W.set_value(W[128:, :, :, :])
                my_net['conv2_part2'].b.set_value(b[128:])
            elif name == 'conv4':
                W = layers_caffe[name].blobs[0].data[:, :, ::-1, ::-1]
                b = layers_caffe[name].blobs[1].data

                my_net['conv4_part1'].W.set_value(W[0:192, :, :, :])
                my_net['conv4_part1'].b.set_value(b[0:192])
                my_net['conv4_part2'].W.set_value(W[192:, :, :, :])
                my_net['conv4_part2'].b.set_value(b[192:])
            elif name == 'conv5':
                W = layers_caffe[name].blobs[0].data[:, :, ::-1, ::-1]
                b = layers_caffe[name].blobs[1].data

                my_net['conv5_part1'].W.set_value(W[0:128, :, :, :])
                my_net['conv5_part1'].b.set_value(b[0:128])
                my_net['conv5_part2'].W.set_value(W[128:, :, :, :])
                my_net['conv5_part2'].b.set_value(b[128:])
            elif name == 'fc6' or name == 'fc7' or name == 'fc8':
                # no need to flip for fully connected layers
                layer.W.set_value(np.transpose(layers_caffe[name].blobs[0].data))
                layer.b.set_value(layers_caffe[name].blobs[1].data)
            else:
                # need to flip to get the same answer in convolution
                layer.W.set_value(layers_caffe[name].blobs[0].data[:, :, ::-1, ::-1])
                layer.b.set_value(layers_caffe[name].blobs[1].data)
        except AttributeError:
            continue
        except KeyError:
            continue

    for f in listdir('../crops/words'):
        im = np.asarray(Image.open('../crops/words/' + f))
        im = im.transpose((2, 0, 1)).reshape(1, 3, im.shape[0], im.shape[1])

        print im.shape

            # run lasagne
        print "Computing lasagne result..."
        lasagne_result = np.array(lasagne.layers.get_output(my_net['pool5'],
                                                            floatX(im),
                                                            deterministic=True).eval())
        print lasagne_result.shape

if __name__ == "__main__":
    load_caffe_weights()
