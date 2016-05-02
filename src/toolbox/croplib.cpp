/*
 * PPM, PGM and PBM image cropping
 * Axel Brink, April 2007
 */

#include "croplib.h"
#include <iostream>
#include <string>
#include <math.h>

// swap two variable values
void swap(long &val1, long &val2)
{
    long dummy = val1;
    val1 = val2;
    val2 = dummy;
}

// crop an image
PamImage* crop(PamImage *im_in, long left, long top, long right, long bottom)
{
    long sx, sy;         // x,y in the source image
    long width = im_in->getWidth();
    long height = im_in->getHeight();
    
    // try to transform the rectangle such that it is inside the image 
    if (left > right) swap(left, right);
    if (top > bottom) swap(top, bottom);
    if (left < 0) left = 0;
    if (right > width - 1) right = width - 1;
    if (top < 0) top = 0;
    if (bottom > height - 1) bottom = height - 1;

    if ((right < 0) || (left > width - 1) || (top > height - 1) || (bottom < 0))
    {
        std::cerr << "Error: cropping rectangle entirely out of image" << std::endl;
        //PamImage* im_out = new PamImage(NO_IMAGE, 0, 0);
        //return im_out;
        throw 1;
        return NULL;
    }
    else
    {
        // cropping rectangle is ok
        long cropwidth = right - left + 1;
        long cropheight = bottom - top + 1;
        int imType = im_in->getImageType();
        if (imType == RGB_IMAGE)
        {
            PamImage* im_out = new PamImage(RGB_IMAGE, cropwidth, cropheight);
            RGBPixel **pixelsIn = im_in->getRGBPixels();
            RGBPixel **pixelsOut = im_out->getRGBPixels();
            for (long dy = 0; dy < cropheight; ++dy)
            {
                sy = top + dy;
                for (long dx = 0; dx < cropwidth; ++ dx)
                {
                    sx = left + dx;
                    pixelsOut[dy][dx].r = pixelsIn[sy][sx].r;
                    pixelsOut[dy][dx].g = pixelsIn[sy][sx].g;
                    pixelsOut[dy][dx].b = pixelsIn[sy][sx].b;
                    pixelsOut[dy][dx].m = pixelsIn[sy][sx].m;
                }
            }
            return im_out;
        }
        else if ((imType == GRAY_IMAGE) || (imType == BW_IMAGE))
        {
            PamImage* im_out = new PamImage(imType, cropwidth, cropheight);
            GrayPixel **pixelsIn = im_in->getGrayPixels();
            GrayPixel **pixelsOut = im_out->getGrayPixels();
            for (long dy = 0; dy < cropheight; ++dy)
            {
                sy = top + dy;
                for (long dx = 0; dx < cropwidth; ++ dx)
                {
                    sx = left + dx;
                    pixelsOut[dy][dx] = pixelsIn[sy][sx];
                }
            }
            return im_out;
        }
        else
        {
            std::cerr << "Error: don't know how to crop image of type " << imType << std::endl;
            PamImage* im_out = new PamImage(NO_IMAGE, 0, 0);
            return im_out;
        }
    }
}
