import sys
# import C++ libraries
import pamImage, croplib

# check commandline parameters
if len(sys.argv) != 3:
    print "Example program. Crops the image to a smaller region."
    print "Usage: python %s image.ppm outfile.ppm" % sys.argv[0]
    sys.exit(1)

in_file_name = sys.argv[1]
out_file_name = sys.argv[2]

# open image
im = pamImage.PamImage(in_file_name)
width, height = im.getWidth(), im.getHeight()
print "Image width:", width
print "Image height:", height

# crop image
print "Cropping image..."
left, right = width / 3, (width * 2) / 3  # integer calculations
top, bottom = 0, height - 1
cropped_im = croplib.crop(im, left, top, right, bottom)
cropped_im.thisown = True                 # to make Python cleanup the new C++ object afterwards
cropped_im.save(out_file_name)
print "%s saved." % out_file_name
