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

from PIL import Image

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

    for line in lines:
        for word in line:
            cropped = img.crop((word.left, word.top, word.right, word.bottom))  # croplib.crop(img, word.left, word.top, word.right, word.bottom)
            out_path = out_str + '_' + word.text + IM_EXT
            cropped.save(out_path)

            for char in word.characters:
                cropped = img.crop((char.left, char.top, char.right, char.bottom))  # croplib.crop(img, word.left, word.top, word.right, word.bottom)
                out_path = out_str + '_' + char.text + IM_EXT
                try:
                    cropped.save(out_path)
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