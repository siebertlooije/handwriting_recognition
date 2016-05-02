import gtk
import pamImage
import croplib
import wordio
import word

def extractImages(wordfile, imgfile):
    lines = wordio.read(wordfile)
    img = pamImage.PamImage(imgfile)

    line_iter = iter(self.lines)
    cur_line = line_iter.next()
    word_iter = iter(cur_line)

    


if __name__ == "__main__":
