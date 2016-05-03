import gtk
import pamImage
import croplib
import wordio
import word as wrd
import os.path as pth

from os import listdir

IM_EXT='.ppm'

import re
import numpy

from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from PIL import Image

import numpy as np

word_log = {}

def extractImages(wordfile, imgfile):
    print wordfile
    lines, _ = wordio.read(wordfile)
    img = Image.open(imgfile)
    #img = pamImage.PamImage(imgfile)

    #line_iter = iter(lines)
    #cur_line = line_iter.next()
    #word_iter = iter(cur_line)

    assert pth.exists('../crops/')
    out_str = '../crops/' + pth.basename(imgfile)
    out_str = out_str.replace(IM_EXT, '')

    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line):
            cropped = img.crop((word.left, word.top, word.right, word.bottom))  # croplib.crop(img, word.left, word.top, word.right, word.bottom)

            out_path = out_str + '_l' + str(line_idx) + '_w' + str(word_idx) + '_t_' + word.text + IM_EXT
            cropped.save(out_path)

            for idx, char in enumerate(word.characters):
                cropped = img.crop((char.left, char.top, char.right, char.bottom))  # croplib.crop(img, word.left, word.top, word.right, word.bottom)

                if 0 in cropped.size:
                    continue
                #print cropped.size
                otsu_img = img_as_ubyte(np.copy(np.asarray(cropped.convert('L'))))

                try:
                    threshold_global_otsu = threshold_otsu(otsu_img)
                except TypeError:
                    print 'Something weird happened'
                    continue
                global_otsu = otsu_img >= threshold_global_otsu

                l_otsu_im = 1 - global_otsu

                index_row = np.tile(np.arange(otsu_img.shape[0]).reshape(otsu_img.shape[0], 1), (1, otsu_img.shape[1]))
                index_col = np.tile(np.arange(otsu_img.shape[1]), (otsu_img.shape[0], 1))

                non0 = l_otsu_im.nonzero()
                Wrow = np.multiply(l_otsu_im[non0], index_row[non0])
                Wcol = np.multiply(l_otsu_im[non0], index_col[non0])

                Mrow = np.mean(Wrow)
                Mcol = np.mean(Wcol)

                Std_row = np.std(Wrow)
                Std_col = np.std(Wcol)

                top = max(0, int(Mrow - 3 * Std_row))
                bottom = min(cropped.size[1], int(Mrow + 3 * Std_row))

                left = max(0, int(Mcol - 3 * Std_col))
                right = min(cropped.size[0], int(Mcol + 3 * Std_col))

                sub_cropped = cropped.crop((left, top, right, bottom))

                out_path = out_str + '_l' + str(line_idx) + '_w' + str(word_idx) + '_c' + str(idx) + '_t_' + char.text + IM_EXT
                try:
                    sub_cropped.save(out_path)
                except SystemError:
                    print 'bad annotation...'

                if char.text in word_log:
                    word_log[char.text] += 1
                else:
                    word_log[char.text] = 1


if __name__ == "__main__":

    print 'Cropping...'
    assert pth.exists('../charannotations/KNMP') and pth.exists('../charannotations/Stanford')

    for f in listdir('../charannotations/KNMP'):
        wordfile = '../charannotations/KNMP/' + f
        imgfile = wordfile.replace('.words', IM_EXT).replace('/charannotations', '/pages').replace('2C20', '2C2O')

        extractImages(wordfile, imgfile)

    for f in listdir('../charannotations/Stanford'):
        wordfile = '../charannotations/Stanford/' + f
        imgfile = wordfile.replace('.words', IM_EXT).replace('/charannotations', '/pages').replace('2C20', '2C2O')



        extractImages(wordfile, imgfile)
    print 'done cropping'
    print 'overview of words: '

    for key in word_log:
        print key, ':', word_log[key]
    #print word_log