#include <iostream>
#include <string.h>

#include "arr1d.h"
#include "arr2d.h"
#include "error.h"

// Constructors and destructor
Array2D::Array2D() {
  arr = 0;
  reSize(0, 0);
}

Array2D::Array2D(int w, int h) {
  arr = 0;
  reSize(w, h);
}

Array2D::Array2D(const Array2D &other) {
  arr = 0;
  reSize(other.wdth, other.hght);
  copy(other);
}

Array2D::~Array2D() {
  deallocate();
}

void Array2D::deallocate() {
  if (arr) {
    delete [] arr[0];
    delete [] arr;
  }
  arr  = 0;
  wdth = 0;
  hght = 0;
}

void Array2D::reSize (int w, int h) {
  // Check arguments
  if (w < 0) {
    error ("Array2D::reSize", "width should be >= 0!\n");
  }
  if (h < 0) {
    error ("Array2D::reSize", "height should be >= 0!\n");
  }

  // allocate memory (if necessary)
  if ((!arr) || (w!=wdth) || (h!=hght)) { 
    // AM: I used explicit left to right evaluation here!
    if (arr) {
      deallocate();
    }
    if ((w==0) || (h==0)) {
      arr  = 0;
    } else {
      arr = new int* [h];
      if (arr == 0) {
	error ("Array2D::reSize", "new(alloc) failed !\n");
      }
      arr[0] = new int [w*h];
      if (arr[0] == 0) {
	error ("Array2D::reSize", "new(alloc) failed !\n");
      }
      for (int i=1; i<h; ++i) {
	arr[i] = arr[i-1] + w;
      }
    }
  }

  // set size fields
  wdth = w;
  hght = h;
}

void Array2D::copy(const Array2D &other) {
  reSize(other.wdth, other.hght);
  memcpy(arr, other.arr, hght*sizeof(int *));
  memcpy(arr[0], other.arr[0], wdth*hght*sizeof(int));
}

void Array2D::setValue (int value) {
  int sz = wdth*hght;
  int *ptr = arr[0];
  for (int i=0; i<sz; ++i) {
    ptr[i] = value;
  }
}

