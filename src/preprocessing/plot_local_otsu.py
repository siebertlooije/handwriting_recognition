"""
====================
Local Otsu Threshold
====================

This example shows how Otsu's threshold [1]_ method can be applied locally. For
each pixel, an "optimal" threshold is determined by maximizing the variance
between two classes of pixels of the local neighborhood defined by a
structuring element.

The example compares the local threshold with the global threshold.

.. note: local is much slower than global thresholding

.. [1] http://en.wikipedia.org/wiki/Otsu's_method

"""

from skimage import data
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

myim = Image.open('../crops/KNMP-VIII_F_69______2C2O_0006_o.ppm').convert('L')

print np.array(myim).shape


matplotlib.rcParams['font.size'] = 9
img = img_as_ubyte(np.copy(np.asarray(myim)))


index_row = np.tile(np.arange(img.shape[0]).reshape(img.shape[0], 1), (1, img.shape[1]))
index_col = np.tile(np.arange(img.shape[1]), (img.shape[0], 1))

print index_row.shape
print index_col.shape



radius = 15
selem = disk(radius)

local_otsu = rank.otsu(img, selem)
threshold_global_otsu = threshold_otsu(img)
global_otsu = img >= threshold_global_otsu

fig, ax = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})
ax0, ax1, ax2, ax3 = ax.ravel()
plt.tight_layout()

fig.colorbar(ax0.imshow(img, cmap=plt.cm.gray),
             ax=ax0, orientation='horizontal')
ax0.set_title('Original')
ax0.axis('off')

fig.colorbar(ax1.imshow(local_otsu, cmap=plt.cm.gray),
             ax=ax1, orientation='horizontal')
ax1.set_title('Local Otsu (radius=%d)' % radius)
ax1.axis('off')


l_otsu_im = 1 - global_otsu

non0 = l_otsu_im.nonzero()
Wrow = np.multiply(l_otsu_im[non0], index_row[non0])
Wcol = np.multiply(l_otsu_im[non0], index_col[non0])

Mrow = np.mean(Wrow)
Mcol = np.mean(Wcol)

Std_row = np.std(Wrow)
Std_col = np.std(Wcol)

print 'M row:', Mrow, 'M col:', Mcol
print 'std row', Std_row, 'std col', Std_col

ax2.imshow(1 - (img >= local_otsu), cmap=plt.cm.gray)
ax2.set_title('Original >= Local Otsu' % threshold_global_otsu)
ax2.axis('off')

ax3.imshow(1 - global_otsu, cmap=plt.cm.gray)
ax3.set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
ax3.axis('off')


plt.show()
