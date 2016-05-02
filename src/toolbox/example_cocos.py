import sys
import pamImage, cocoslib

def coco_crop(in_fname, out_fname):
    im = pamImage.PamImage(in_fname)
    # Load object that finds the cocos
    # The parameters specify 8-connectedness
    # and a foreground color of 0 (the ink should be black)
    cocos_thing = cocoslib.Cocos(im, 8, 0)
    num = cocos_thing.getNum()
    print "%i black connected components found." % num
    if num > 0:
        ### Get the coordinates of the first coco's bounding box
        left, top, right, bottom = cocos_thing.getCocoRect(0)
        print "The first coco is located at (%i, %i)-(%i, %i)" % \
              (left, top, right, bottom)
        ### Make a small image of the first coco ###
        coco_im = cocos_thing.getCocoIm(0)
        coco_im.thisown = True # Make Python the boss of this C++ object
                               # (This avoids a memory leak)
        coco_im.save(out_fname)
        print "%s saved." % out_fname

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Crop a black connected component from a binary (black-and-white) image."
        print "Usage: python %s input.pgm output.pgm" % (sys.argv[0])
        print "input.pgm must be binary!"
        print "output.pgm will contain one black connected component."
    else:
        in_fname = sys.argv[1]
        out_fname = sys.argv[2]
        coco_crop(in_fname, out_fname)
