#include <fstream>
#include <string.h>
#include <iostream>

#include "cocoslib.h"

// to be replaced with file convert_array2d.cpp / .h
void _convertToArray2D(PamImage *im, Array2D &im2d)
{
    int imageType = im->getImageType();
    if ((imageType != BW_IMAGE) && (imageType != GRAY_IMAGE))
    {
        std::cerr << "Error: illegal image type for conversion to Array2D: " << imageType << std::endl;
        throw 1;
    }

    long width = im->getWidth();
    long height = im->getHeight();
    GrayPixel **pixels_in = im->getGrayPixels();
    
    im2d.reSize(width, height);

    for (long y = 0; y < height; y++) {
        for (long x = 0; x < width; x++) {
            im2d[y][x] = (int) pixels_in[y][x];
        }
    }
}
    
PamImage *_convertToPamImage(Array2D &im2d)
{
    long width = im2d.width();
    long height = im2d.height();
    PamImage* im_out = new PamImage(INT_IMAGE, width, height);
    IntPixel **pixels_out = im_out->getIntPixels();

    for (long y = 0; y < height; y++) {
        for (long x = 0; x < width; x++) {
            pixels_out[y][x] = im2d[y][x];
        }
    }
    return im_out;
}

//// Old
//PamImage* cocos4(PamImage *im) {
//    Array2D im2d;
//    _convertToArray2D(im, im2d);
//    componentLabeling4(im2d);
//    //int count = componentLabeling4(im2d);
//    PamImage* im_out = _convertToPamImage(im2d);
//    return im_out;
//}
//
//// Old
//PamImage* cocos8(PamImage *im) {
//    Array2D im2d;
//    _convertToArray2D(im, im2d);
//    componentLabeling8(im2d);
//    //int count = componentLabeling8(im2d);
//    PamImage* im_out = _convertToPamImage(im2d);
//    return im_out;
//}

/* Label each component with an int */
/*
PamImage* cocos(PamImage *im) {
    Array2D im2d;
    int count;                    // number of connected components
    long width = im->getWidth();
    long height = im->getHeight();
    GrayPixel **pixels_in = im->getGrayPixels();
    
    im2d.reSize(width, height);

    for (long y = 0; y < height; y++) {
        for (long x = 0; x < width; x++) {
            im2d[y][x] = (int) pixels_in[y][x];
        }
    }

    count = componentLabeling4(im2d);
    //printf("%i components found.\n", count);
    //count = componentLabeling8(im2d);

    PamImage* im_out = new PamImage(INT_IMAGE, width, height);
    IntPixel **pixels_out = im_out->getIntPixels();

    for (long y = 0; y < height; y++) {
        for (long x = 0; x < width; x++) {
            pixels_out[y][x] = im2d[y][x];
        }
    }

    return im_out;
}
*/

// Constructor
// Cocos with color 'backcolor' are removed.
Cocos::Cocos(PamImage* im, int connectivity, int forecolor)
{
    int type = im->getImageType(); 
    if (type != BW_IMAGE && type != GRAY_IMAGE)
    {
        std::cerr << "Error: image is not black/white or grayscale" << std::endl;
        throw 1;
    }
    if (connectivity == 4)
    {
        _convertToArray2D(im, labelAr2D);
        num = componentLabeling4(labelAr2D);
    }
    else if (connectivity == 8)
    {
        _convertToArray2D(im, labelAr2D);
        num = componentLabeling8(labelAr2D);
    }
    else
    {
        std::cerr << "Illegal connectivity: %i. Must be 4 or 8." << std::endl;
        throw 1;
    }
    _makeCocoList(im);
    _removeBackgroundCocos(forecolor);
}

// Destructor
Cocos::~Cocos()
{
    delete [] cocoList;
}

// Get number of cocos
int Cocos::getNum()
{
    return num;
}

// Fill cocoList
void Cocos::_makeCocoList(PamImage* image)
{
    cocoList = new coco[num];
    long width = labelAr2D.width();
    long height = labelAr2D.height();
    int label;
    
    GrayPixel **pixels = image->getGrayPixels();

    for (label = 0; label < num; ++label) {
        cocoList[label].left = width - 1;
        cocoList[label].right = 0;
        cocoList[label].top = height - 1;
        cocoList[label].bottom = 0;
        cocoList[label].label = label;
        cocoList[label].surf = 0;
    }

    for (long y = 0; y < height; y++) {
        GrayPixel *pixelrow = pixels[y];
        for (long x = 0; x < width; x++) {
            label = labelAr2D[y][x];
            coco &item = cocoList[label];
            if (x < item.left) item.left = x;
            if (x > item.right) item.right = x;
            if (y < item.top) item.top = y;
            if (y > item.bottom) item.bottom = y;
            item.grayval = pixelrow[x];
            item.surf++;
        }
    }
    
}

// Remove cocos with color other than foreground. Changes coco ordering in cocoList.
void Cocos::_removeBackgroundCocos(int forecolor)
{
    int label = 0;
    while (label < num)
    {
        if (cocoList[label].grayval != forecolor)
        {
            // remove this item: replace by last item
            cocoList[label] = cocoList[num - 1];
            num--;
        }
        else
        {
            label++;
        }
    }
}

PamImage* Cocos::getCocoIm(int label)
{
    if ((label < 0) || (label > num - 1))
    {
        std::cerr << "Error: label index %ld out of bounds %ld..%ld" << std::endl;
        throw 1;
    } 
    coco item = cocoList[label];
    int realLabel = item.label; // label in labelAr2D
    long width = item.right - item.left + 1;
    long height = item.bottom - item.top + 1;
    long sx, sy; // coordinates in source image
    PamImage* im_out = new PamImage(BW_IMAGE, width, height);
    GrayPixel **pixels_out = im_out->getGrayPixels();
    //std::cerr << "Looking for label: " << realLabel << std::endl;
    for (int y = 0; y < height; ++y)
    {
        sy = y + item.top;
        for (int x = 0; x < width; ++x)
        {
            sx = x + item.left;
            //std::cerr << labelAr2D[sy][sx] << " ";
            if (labelAr2D[sy][sx] != realLabel)
            {
                pixels_out[y][x] = 1;
            }
        }
        //std::cerr << std::endl;
        
    }
    return im_out;
}

void Cocos::getCocoRect(int label, long &left, long &top, long &right, long &bottom)
{
    if ((label < 0) || (label > num - 1))
    {
        std::cerr << "Error: label index %ld out of bounds %ld..%ld" << std::endl;
        throw 1;
    } 
    coco item = cocoList[label];
    left = item.left;
    right = item.right;
    top = item.top;
    bottom = item.bottom;
}

PamImage* Cocos::getCocosIm()
{
    return _convertToPamImage(labelAr2D);
}

// Gray image with black=foreground (coco); white=background
// Drawing is based on cocoList
PamImage* Cocos::getImage()
{
    long width = labelAr2D.width();
    long height = labelAr2D.height();
    PamImage* im_out = new PamImage(GRAY_IMAGE, width, height);
    GrayPixel **pixels_out = im_out->getGrayPixels();
    
    for (long y = 0; y < height; y++) {
        for (long x = 0; x < width; x++) {
            pixels_out[y][x] = 255; //labelAr2D[y][x];
        }
    }

    for (int label = 0; label < num; ++label) {
        coco &item = cocoList[label];
        for (int y = item.top; y <= item.bottom; ++y) {
            for (int x = item.left; x <= item.right; ++x) {
                if (labelAr2D[y][x] == item.label) {
                    pixels_out[y][x] = 0;
                }
            }
        }
    }
    
    return im_out;
}

long Cocos::getSurface(int label)
{
    return cocoList[label].surf;
}

// Remove a coco from the list. It will not be drawn. Changes coco ordering.
void Cocos::remove(int label)
{
    if (label < 0 || label >= num) {
        std::cerr << "Error: cannot remove connected component label with label " << label
                  << "; outside range [0.." << num << ">. (cocoslib.cpp)" << std::endl;
    } else { 
        cocoList[label] = cocoList[num - 1];
        num--;
    }
}
