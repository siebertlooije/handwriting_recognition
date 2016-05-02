#include <stdio.h>

#include "arr2d.h"

#define MAX(a,b) ((a)>(b) ? (a) : (b))
#define MIN(a,b) ((a)<(b) ? (a) : (b))

static int findRoot(int *par, int x) {
  while (par[x] != x) {
    x = par[x];
  }
  return x;
}

static void pathCompress (int *par, int x, int root) {
  int h;
  while (x!=root) {
    h = par[x];
    par[x] = root;
    x = h;
  }
}

static int Link(int *par, int x, int y) {
  int rx = findRoot(par, x);
  int ry = findRoot(par, y);
  int newroot;
  if (rx<ry) {
    par[ry] = rx;
    newroot = rx;
  } else {
    par[rx] = ry;
    newroot = ry;
  }
  pathCompress (par, x, newroot);
  pathCompress (par, y, newroot);
  return newroot;
}

static inline bool equal (int x, int y, int delta) {
  if (x<y) {
    return y-x<=delta;
  }
  return x-y<=delta;
}

static int resolve (int sz, int *par) {
  int cnt = 0;
  int p;
  for (int idx=0; idx<sz; ++idx) {
    p = par[idx];
    if (p == idx)
      par[idx] = cnt++;
    else
      par[idx] = par[p];
  }
  return cnt;
}

static void copyLabeling(Array2D &im, int sz, int *par) {
  int idx = 0;
  int w = im.width();
  int h = im.height();
  for (int i=0; i<h; ++i)
    for (int j=0; j<w; ++j)
      im[i][j] = par[idx++];
}

/**************************** 4 connected ************************/

int componentLabeling4(Array2D &im) {
  int h = im.height();
  int w = im.width();
  int sz=w*h;
  int *par = new int [sz];

  // first pass (horizontal scans)
  int idx=0;
  for (int i=0; i<h; ++i) {
    par[idx] = idx;
    ++idx;
    for (int j=1; j<w; ++j) {
      if (im[i][j] == im[i][j-1]) {
	par[idx] = par[idx-1];
      } else {
	par[idx] = idx;
      }
      ++idx;
    }
  }

  // second pass (vertical scans), bottom-up is most efficient
  for (int i=h-2; i>=0; --i) {
    idx = i*w;
    for (int j=0; j<w; ++j) {
      if  (im[i][j] == im[i+1][j]) {
	// equivalence found
	par[idx] = Link(par, idx, idx+w);
      }
      ++idx;
    }
  }

  // resolving pass
  int cnt = resolve (sz, par);

  // copy to image
  copyLabeling(im, sz, par);

  // free par array
  delete [] par;

  // return number of labels
  return cnt;
}

int componentLabeling4(Array2D &im, int delta) {
  int h = im.height();
  int w = im.width();
  int sz=w*h;
  int *par = new int [sz];

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
  for (int i=h-2; i>=0; --i) {
    idx = i*w;
    for (int j=0; j<w; ++j) {
      if  (equal(im[i][j], im[i+1][j], delta)) {
	// equivalence found
	par[idx] = Link(par, idx, idx+w);
      }
      ++idx;
    }
  }

  // resolving pass
  int cnt = resolve (sz, par);

  // copy to image
  copyLabeling(im, sz, par);

  // free par array
  delete [] par;

  // return number of labels
  return cnt;
}

/**************************** 8 connected ************************/

int componentLabeling8(Array2D &im) {
  int h = im.height();
  int w = im.width();
  int sz=w*h;
  int *par = new int [sz];

  // first pass (horizontal scans)
  int idx=0;
  for (int i=0; i<h; ++i) {
    par[idx] = idx;
    ++idx;
    for (int j=1; j<w; ++j) {
      if (im[i][j] == im[i][j-1]) {
	par[idx] = par[idx-1];
      } else {
	par[idx] = idx;
      }
      ++idx;
    }
  }

  // second pass , bottom-up is most efficient
  for (int i=h-2; i>=0; --i) {
    idx = i*w;
    /* case j==0 (separately) */
    if  (im[i][0] == im[i+1][0]) {
      // equivalence found
      par[idx] = Link(par, idx, idx+w);
    }
    if  (im[i][0] == im[i+1][1]) {
      // equivalence found
      par[idx] = Link(par, idx, idx+w+1);
    }
    idx++;

    for (int j=1; j<w-1; ++j) {
      if  (im[i][j] == im[i+1][j-1]) {
	// equivalence found
	par[idx] = Link(par, idx, idx+w-1);
      }
      if  (im[i][j] == im[i+1][j]) {
	// equivalence found
	par[idx] = Link(par, idx, idx+w);
      }
      if  (im[i][j] == im[i+1][j+1]) {
	// equivalence found
	par[idx] = Link(par, idx, idx+w+1);
      }
      ++idx;
    }
    /* case j==w-1 (separately) */
    if  (im[i][w-1] == im[i+1][w-2]) {
      // equivalence found
      par[idx] = Link(par, idx, idx+w-1);
    }
    if  (im[i][w-1] == im[i+1][w-1]) {
      // equivalence found
      par[idx] = Link(par, idx, idx+w);
    }
    idx++;
  }

  // resolving pass
  int cnt = resolve (sz, par);

  // copy to image
  copyLabeling(im, sz, par);

  // free par array
  delete [] par;

  // return number of labels
  return cnt;
}

int componentLabeling8(Array2D &im, int delta) {
  int h = im.height();
  int w = im.width();
  int sz=w*h;
  int *par = new int [sz];

  // first pass (horizontal scans)
  int idx=0;
  for (int i=0; i<h; ++i) {
    par[idx] = idx;
    ++idx;
    for (int j=1; j<w; ++j) {
      if (equal(im[i][j],im[i][j-1],delta)) {
	par[idx] = par[idx-1];
      } else {
	par[idx] = idx;
      }
      ++idx;
    }
  }

  // second pass , bottom-up is most efficient
  for (int i=h-2; i>=0; --i) {
    idx = i*w;
    /* case j==0 (separately) */
    if  (equal(im[i][0],im[i+1][0],delta)) {
      // equivalence found
      par[idx] = Link(par, idx, idx+w);
    }
    if  (equal(im[i][0],im[i+1][1],delta)) {
      // equivalence found
      par[idx] = Link(par, idx, idx+w+1);
    }
    idx++;

    for (int j=1; j<w-1; ++j) {
      if  (equal(im[i][j],im[i+1][j-1],delta)) {
	// equivalence found
	par[idx] = Link(par, idx, idx+w-1);
      }
      if  (equal(im[i][j],im[i+1][j],delta)) {
	// equivalence found
	par[idx] = Link(par, idx, idx+w);
      }
      if  (equal(im[i][j],im[i+1][j+1],delta)) {
	// equivalence found
	par[idx] = Link(par, idx, idx+w+1);
      }
      ++idx;
    }
    /* case j==w-1 (separately) */
    if  (equal(im[i][w-1],im[i+1][w-2],delta)) {
      // equivalence found
      par[idx] = Link(par, idx, idx+w-1);
    }
    if  (equal(im[i][w-1],im[i+1][w-1],delta)) {
      // equivalence found
      par[idx] = Link(par, idx, idx+w);
    }
    idx++;
  }

  // resolving pass
  int cnt = resolve (sz, par);

  // copy to image
  copyLabeling(im, sz, par);

  // free par array
  delete [] par;

  // return number of labels
  return cnt;
}

/**************************** any connectivity ************************/

int componentLabeling(int nnb, int nb[][2], Array2D &im) {
  int h = im.height();
  int w = im.width();
  int sz=w*h;
  int *par = new int [sz];
  int i0, i1, j0, j1, di, dj;
  int idx, offset;
  int i, j, ii, jj;

  for (i=0; i<sz; ++i) 
    par[i] = i;

  for (int n=0; n<nnb; ++n) {
    /* check for symmetry vector */
    for (i=0; i<n; ++i) {
      if ((nb[i][0] == nb[n][0]) && (nb[i][1] == nb[n][1]))
	break;
      if ((nb[i][0] == -nb[n][0]) && (nb[i][1] == -nb[n][1]))
	break;
    }
    if (i<n) continue;

    /* process neighbour vector n */
    di = nb[n][0];
    dj = nb[n][1];
    i0 = MAX(0, -di); i1=MIN(h,h-di);
    j0 = MAX(0, -dj); j1=MIN(w,w-dj);
    offset = di*w + dj;

    for (i=i0, ii=i0+di; i<i1; ++i,++ii) {
      idx = i*w + j0;
      for (j=j0,jj=j0+dj; j<j1; ++j,++jj) {
	if (im[i][j] == im[ii][jj]) {
	  par[idx] = Link(par, idx, idx+offset);
	}
	++idx;
      }
    }
  }
  // resolving pass
  int cnt = resolve (sz, par);

  // copy to image
  copyLabeling(im, sz, par);

  // free par array
  delete [] par;

  // return number of labels
  return cnt;
}


int componentLabeling(int nnb, int nb[][2], Array2D &im, int delta) {
  int h = im.height();
  int w = im.width();
  int sz=w*h;
  int *par = new int [sz];
  int i0, i1, j0, j1, di, dj;
  int idx, offset;
  int i, j, ii, jj;

  for (i=0; i<sz; ++i) 
    par[i] = i;

  for (int n=0; n<nnb; ++n) {
    /* check for symmetry vector */
    for (i=0; i<n; ++i) {
      if ((nb[i][0] == nb[n][0]) && (nb[i][1] == nb[n][1]))
	break;
      if ((nb[i][0] == -nb[n][0]) && (nb[i][1] == -nb[n][1]))
	break;
    }
    if (i<n) continue;

    /* process neighbour vector n */
    di = nb[n][0];
    dj = nb[n][1];
    i0 = MAX(0, -di); i1=MIN(h,h-di);
    j0 = MAX(0, -dj); j1=MIN(w,w-dj);
    offset = di*w + dj;

    for (i=i0, ii=i0+di; i<i1; ++i,++ii) {
      idx = i*w + j0;
      for (j=j0,jj=j0+dj; j<j1; ++j,++jj) {
	if (equal(im[i][j], im[ii][jj], delta)) {
	  par[idx] = Link(par, idx, idx+offset);
	}
	++idx;
      }
    }
  }
  // resolving pass
  int cnt = resolve (sz, par);

  // copy to image
  copyLabeling(im, sz, par);

  // free par array
  delete [] par;

  // return number of labels
  return cnt;
}

