#ifndef __CROPLIB_H__
#define __CROPLIB_H__
 
#include "pamImage.h"

PamImage* crop(PamImage *im_in, long left, long top, long right, long bottom);
 
#endif
