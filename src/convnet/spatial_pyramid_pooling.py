
import theano.tensor as T
import theano.tensor.signal.pool as pool
import numpy as np


class SPP(object):

    def __init__(self, input, pooling_factors=[4, 2, 1]):
        '''
        This constructs a Spatial Pyramid Pooling layer from a list of pooling factors. For example, a
        pooling factor of 4 will return a 4 x 4 set of spatial bins after pooling. For each pooling factor
        these spatial bins are flattened and then concatenated to form the layers output. Note  that the
        layer's output size is independent of the input size (needs to be at least 4x4).
        :param input: A 4-D tensor, the last 2 dimensions are pooled
        :param pooling_factors: List of pooling factors
        '''
        spatial_bins = []

        # This 'M' corresponds to the M as seen in Kaiming et al. which denotes the number of spatial
        # bins for a single kernel
        self.M = np.sum(np.asarray(pooling_factors) ** 2)

        for fac in pooling_factors:
            spatial_bins.append(pool.pool_2d(input, (fac, fac), ignore_border=True))

        self.output = T.concatenate([sb.flatten(2) for sb in spatial_bins])



