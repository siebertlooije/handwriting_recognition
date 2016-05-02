#include "ailib.h"

int threshold(int th, Array2D &im) {
  int w = im.width();
  int h = im.height();
  int cnt=0;
  for (int i=0; i<h; ++i) {
    for (int j=0; j<w; ++j) {
      if (im[i][j] < th) {
	im[i][j] = 0;
      } else {
	im[i][j] = 255;
	++cnt;
      }
    }
  }
  return cnt;
}
