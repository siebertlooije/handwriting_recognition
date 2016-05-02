#include <stdio.h>

#include "arr2d.h"

static int Link(int *par, int x, int y) {
  // invariant: x >= y
  int h;
  int x0, y0;

  x0 = x;
  y0 = y;

  if (x<y) {
    // swap x, y
    h = x;
    x = y;
    y = h;
  }
  while ((x!=y) && (par[x] != x)) {
    x = par[x];
    if (x<y) {
      // swap x, y
      h = x;
      x = y;
      y = h;
    }
  }

  if (x != y) {
    par[x] = y;
  }

  // path compression
  x = x0;
  while (x!=y) {
    h = par[x];
    par[x] = y;
    x = h;
  }

  x = y0;
  while (x!=y) {
    h = par[x];
    par[x] = y;
    x = h;
  }
  return y;
}

static inline bool equal (int x, int y, int delta) {
  if (x<y) {
    return y-x<=delta;
  }
  return x-y<=delta;
}

int componentLabeling(int connectivity, Array2D &im, int delta) {
  int h = im.height();
  int w = im.width();
  int *par = new int [w*h];

  // first pass (horizontal scans)
  int idx=0;
  for (int i=0; i<h; ++i) {
    par[idx] = idx;
    ++idx;
    for (int j=1; j<w; ++j) {
      if (equal(im[i][j], im[i][j-1], delta)) {
	par[idx] = par[idx-1];
      } else {
	par[idx] = idx;
      }
      ++idx;
    }
  }

  // second pass (vertical scans), bottom-up is most efficient
  --idx;
  for (int i=h-1; i>0; --i) { /* note i>0, not i>=0 ! */
    for (int j=w-1; j>=0; --j) { /* note j>=0, not j >0 */
      if (equal(im[i][j], im[i-1][j], delta)) {
	// equivalence found
	par[idx] = Link(par, idx, idx-w);
      }
      --idx;
    }
  }
  // resolving pass
  int cnt = 0;
  idx = 0;
  for (int i=0; i<h; ++i) {
    for (int j=0; j<w; ++j,++idx) {
      if (par[idx]==idx) {
	im[i][j] = cnt;
	par[idx] = cnt;
	cnt++;
      } else {
	par[idx] = par[par[idx]];
	im[i][j] = par[idx];
      }
    }
  }
  delete [] par;
  // return number of labels
  return cnt;
}

int componentLabeling(int connectivity, Array2D &im) {
  return componentLabeling(connectivity, im, 0);
}
