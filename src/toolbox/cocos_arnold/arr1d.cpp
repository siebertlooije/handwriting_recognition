#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "arr1d.h"

static void error (char *funcname, char *errmsg) {
  std::cerr << funcname << ": " << errmsg << "\n";
  exit(EXIT_FAILURE);
}

// Constructors and destructor
Array1D::Array1D() {
  arr = 0;
  setSize(0);
}

Array1D::Array1D(int length) {
  arr = 0;
  setSize(length);
}

Array1D::Array1D(const Array1D &other) {
  arr = 0;
  setSize(other.len);
  copy(other);
}

Array1D::~Array1D() {
  deallocate();
}

void Array1D::deallocate() {
  if (arr) {
    arr -= x0;
    delete [] arr;
  }
  arr = 0;
  len = 0;
}

void Array1D::setSize (int length) {
  // Check arguments
  if (length < 0) {
    error ("Array1D::setSize", "lerngth should be >= 0!\n");
  }

  // allocate memory (if necessary)
  if ((!arr) || (len!=length)) { // AM: I used explicit left to right evaluation here!
    if (arr) {
      deallocate();
    }
    if (length==0) {
      arr  = 0;
    } else {
      arr = new int [length];
    }
  }

  // set size fields
  len = length;
  x0 = 0;
}

void Array1D::copy(const Array1D &other) {
  memcpy(arr, other.arr, len*sizeof(int));
}

void Array1D::setValue (int value) {
  for (int i=0; i<len; ++i) {
    arr[i] = value;
  }
}


