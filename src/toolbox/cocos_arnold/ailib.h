#ifndef AILIB_H
#define AILIB_H

#include "arr2d.h"

/* PGM image I/O */
void readImage (char *fileName, Array2D &im);
void writeImage (char *fileName, Array2D &im);

/* (fuzzy) connected component labeling, return the number of cc's */
int componentLabeling4(Array2D &im);
int componentLabeling4(Array2D &im, int delta);
int componentLabeling8(Array2D &im);
int componentLabeling8(Array2D &im, int delta);
int componentLabeling(int nnb, int nb[][2], Array2D &im);
int componentLabeling(int nnb, int nb[][2], Array2D &im, int delta);

/* thresholding */
int threshold(int th, Array2D &im);


#endif
