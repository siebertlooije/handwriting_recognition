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
from scipy.signal import argrelextrema


class Monogram(object):

    def __init__(self, im, start, end):
        self.im = im
        self.start = start
        self.end = end


def augment(myim):
    meanInk = np.zeros(3)
    meanPaper = np.ones(3) * 255
    def assign_points():
        distInk = myim - meanInk
        distPap = myim - meanPaper

        distInk **= 2
        distPap **= 2

        return np.sum(distInk, axis=2) > np.sum(distPap, axis=2)

    points = assign_points()
    mask = np.tile(points, (3, 1, 1)).transpose((1, 2, 0))
    augmented = np.copy(myim)
    augmented[mask] += np.min(255 - augmented[mask])
    augmented[mask == 0] -= np.min(augmented[mask == 0])
    augmented = np.minimum(augmented, 255)
    augmented = np.maximum(augmented, 0)
    return augmented

def get_monograms(img):
    hist = np.mean(img, axis=(0, 2))

    hist = np.maximum(np.mean(img[img.shape[0] / 2:], axis=(0, 2)), hist)
    hist = np.maximum(np.mean(img[:img.shape[0] / 2], axis=(0, 2)), hist)

    plt.imshow(img)
    smooth_win = 20

    hist_small = np.convolve(hist, 5 * [1 / 5.], 'same') / 25

    hist = np.convolve(hist, 11 * [1 / 11.], 'same') / 25

    hist[:smooth_win - 1] = hist_small[:smooth_win - 1]
    hist[-(smooth_win - 1):] = hist_small[-(smooth_win - 1):]
    maxes = argrelextrema(hist, np.greater)

    cuts = maxes[0]
    for idx in range(len(cuts) - 1):
        while len(cuts) > idx + 1 and cuts[idx + 1] - cuts[idx] < 10:
            cuts[idx] = cuts[idx] if hist[cuts[idx]] > hist[cuts[idx + 1]] else cuts[idx]
            cuts = np.delete(cuts, idx + 1)

    monograms = []

    for idx, cut in enumerate(cuts):
        for next in cuts[idx + 1:]:
            print next, cut
            if 10 < next - cut < 50:
                monograms.append(Monogram(img[:, cut:next], cut, next))

    return monograms

def show_hists(wordfile, imgfile):
    lines, _ = wordio.read(wordfile)
    img = Image.open(imgfile)

    for line_idx, line in enumerate(lines):
        for word_idx, word in enumerate(line):
            cropped = img.crop((word.left, word.top, word.right, word.bottom))
            color = np.copy(np.asarray(cropped))

            print color.shape
            #color = augment(color)
            hist = np.mean(color, axis=(0,2))

            hist = np.maximum(np.mean(color[color.shape[0]/2:], axis=(0, 2)), hist)
            hist = np.maximum(np.mean(color[:color.shape[0] / 2], axis=(0, 2)), hist)
            hist = np.maximum(np.mean(color[color.shape[0]/4:-color.shape[0]/4], axis=(0, 2)), hist)
            hist = np.maximum(np.mean(color[color.shape[0]/8:-3*color.shape[0]/8], axis=(0, 2)), hist)
            hist = np.maximum(np.mean(color[3*color.shape[0]/8:-color.shape[0]/8], axis=(0, 2)), hist)

            print hist.shape

            otsu_img = img_as_ubyte(np.copy(np.asarray(cropped.convert('L'))))

            try:
                threshold_global_otsu = threshold_otsu(otsu_img)
            except TypeError:
                print 'Something weird happened'
                continue
            global_otsu = np.array(otsu_img >= threshold_global_otsu).astype(np.int64)

            hist2 = np.zeros(global_otsu.shape[1])
            for col in range(global_otsu.shape[1]):
                max_white = 0
                white = 0
                for row in range(global_otsu.shape[0]):
                    white += 1 if global_otsu[row, col] == 1 else 0
                    max_white = max(white, max_white)
                hist2[col] = max_white

            plt.imshow(color)
            smooth_win = 11

            hist = hist2

            #hist_small = np.convolve(hist, 5 * [1 / 5.], 'same')

            hist = np.convolve(hist, 5 * [1/5.], 'same') / 25

            #hist[:smooth_win-1] = hist_small[:smooth_win-1]
            #hist[-(smooth_win-1):] = hist_small[-(smooth_win-1):]
            plt.bar(np.arange(color.shape[1]), hist)
            plt.title(word.text)

            for idx in range(1, len(hist)-1):
                if hist[idx] == hist[idx+1] and hist[idx-1] < hist[idx]:
                    hist[idx] = (hist[idx+1] + hist[idx-1]) / 2

            maxes = argrelextrema(hist, np.greater)

            med = np.median(hist)

            cuts = maxes[0]
            '''
            for idx in range(len(cuts) - 1):
                #while len(cuts) > idx + 1 and hist[cuts[idx]] < med:
                #    cuts = np.delete(cuts, idx)

                while len(cuts) > idx + 1 and cuts[idx + 1] - cuts[idx] < 10:
                    cuts[idx] = cuts[idx] if hist[cuts[idx]] > hist[cuts[idx+1]] else cuts[idx]
                    cuts = np.delete(cuts, idx+1)
            '''

            plt.bar(cuts, len(cuts) * [color.shape[0]], color='r')

            print hist

            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

            plt.show()

            monograms = []


            for idx, cut in enumerate(cuts):
                for next in cuts[idx+1:]:
                    print next, cut
                    if 10 < next - cut < 50:
                        monograms.append(Monogram(color[:, cut:next], cut, next))

            #for gram in monograms:
            #    plt.imshow(gram.im)
            #    plt.show()




if __name__ == "__main__":

    label_file = open('labels.txt', 'w')


    print 'Cropping...'
    assert pth.exists('../charannotations/KNMP') and pth.exists('../charannotations/Stanford')

    for f in listdir('../charannotations/Stanford'):
        wordfile = '../charannotations/Stanford/' + f
        imgfile = wordfile.replace('.words', IM_EXT).replace('/charannotations', '/pages').replace('2C20', '2C2O')

        show_hists(wordfile, imgfile)