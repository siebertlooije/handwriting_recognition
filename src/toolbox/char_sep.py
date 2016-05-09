import gtk
import pamImage
import croplib
import wordio
import word as wrd
import os.path as pth

from os import listdir

IM_EXT='.ppm'


from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

def show_hists(wordfile, imgfile):
    lines, _ = wordio.read(wordfile)
    img = Image.open(imgfile)

    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line):
            cropped = np.asarray(img.crop((word.left, word.top, word.right, word.bottom)))
            otsu_img = img_as_ubyte(np.copy(np.asarray(cropped.convert('L'))))

            try:
                threshold_global_otsu = threshold_otsu(otsu_img)
            except TypeError:
                print 'Something weird happened'
                continue
            global_otsu = otsu_img >= threshold_global_otsu

            hist = np.mean(global_otsu, axis=0)

            plt.imshow(global_otsu, cmap=plt.cm.gray)
            plt.bar(np.arange(global_otsu.shape[1]), hist)

            plt.show()



if __name__ == "__main__":

    label_file = open('labels.txt', 'w')


    print 'Cropping...'
    assert pth.exists('../charannotations/KNMP') and pth.exists('../charannotations/Stanford')

    for f in listdir('../charannotations/KNMP'):
        wordfile = '../charannotations/KNMP/' + f
        imgfile = wordfile.replace('.words', IM_EXT).replace('/charannotations', '/pages').replace('2C20', '2C2O')

        show_hists(wordfile, imgfile)