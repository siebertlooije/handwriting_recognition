import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import math
label_dict = {'!': 1, '#' : 2, '$' : 3, '&': 4, '*': 5, ',': 6, '-' : 7, '.': 8, '4' : 9, ':' : 10, ';' : 11, '?' : 12,
              '\\' : 13, '^': 14, 'a': 15, 'b': 16, 'c': 17, 'd' : 18, 'e' : 19, 'f' : 20, 'g' : 21, 'h' : 22, 'i' : 23,
              'k' : 24, 'l' : 25, 'm' : 26, 'n' : 27, 'o' : 28, 'p' : 29, 'q' : 30, 'r' : 31, 's' : 32, 't' : 33,
              'u' : 34, 'v' : 35, 'w' : 36, 'x' : 37, 'y' : 38, 'z' : 39}


def to_histogram(string, region):
    hist = np.zeros(39)

    # skip the part that has no overlap
    beg = int(math.floor(region[0]))
    string = string[int(math.floor(region[0])): int(math.ceil(region[1]))]

    for char in string:
        # add only if enough overlap between char occupance and region
        end = beg+1
        if min(end, region[1]) - max(beg, region[0]) >= 0.5:
            hist[label_dict[char]] += 1
        beg = end
    return hist

def string_to_PHOC(word, levels = 5) :
    levelrange = range(1, levels+1)
    PHOC = np.asarray([])

    for level in levelrange:
        reglen = len(word)/float(level)
        regions = [(start, start + reglen) for start in [a * reglen for a in range(0,level)]]
        for region in regions:
            PHOC = np.append(PHOC, to_histogram(word, region))
    return PHOC

def PHOC_to_string(PHOC) :
    pass

if __name__ == "__main__":
    levels = 15
    phoc = string_to_PHOC("grasmaaier", levels)
    print phoc.shape

    noise = np.random.normal(0,0.3, 39*sum(range(1,levels+1)))
    phoc = np.add(noise, phoc)

    binsize = 1000
    lines = np.zeros((39, binsize))

    level_range = range(1, levels+1)
    hnr = 0
    for level in level_range:
        reg_len = 1.0/level
        regions = [(start, start + reg_len) for start in [a * reg_len for a in range(0, level)]]
        for region in regions:
            for char_idx in range(0, 39):
                for i in range(int(round(region[0] * binsize)), int(round(region[1] * binsize))):
                    lines[char_idx, i] += phoc[hnr*39 + char_idx]
            hnr += 1


    for i in range(0,39):
        plt.plot(lines[i,:])
    plt.show()




