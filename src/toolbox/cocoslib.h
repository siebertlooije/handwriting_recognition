#ifndef COCOSLIB_H
#define COCOSLIB_H

#include "ailib.h"
#include "pamImage.h"

class Cocos
{
    private:
        struct coco
        {
            int grayval;
            int label;
            long left, right, top, bottom;
            long surf; // number of pixels in the coco (NIY)
        };
        Array2D labelAr2D;
        coco* cocoList;
        int num;
        void _makeCocoList(PamImage* image);
        void _removeBackgroundCocos(int forecolor);

    public:
        Cocos(PamImage* im, int connectivity, int forecolor);
        ~Cocos();
        int getNum();
        PamImage* getCocoIm(int label);
        void getCocoRect(int label, long &left, long &top, long &right, long &bottom);
        PamImage* getCocosIm(); // Int image with colored labels
        PamImage* getImage();   // Gray image with black=coco; white=background 
        long getSurface(int label);
        void remove(int label); // Remove coco from coco list. Changes ordering.
};

#endif //COCOSLIB_H
