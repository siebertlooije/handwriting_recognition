#ifndef ARR2D_H
#define ARR2D_H

#include "arr1d.h"

class Array2D {
  public:
    // Constructors and destructor
    Array2D();    
    Array2D(int wdth, int hght);
    Array2D(const Array2D &other);
    ~Array2D();  

    // operators
    const int *operator[](int index) const { return arr[index]; };
    int *operator[](int index) { return arr[index]; };
     
    // misc. methods
    int  width() { return wdth; }
    int  height() { return hght; }
    void copy(const Array2D &other);
    void reSize(int wdth, int hght);
    void setValue(int value);
    void deallocate();
    int rank(int i, int j) { return i*wdth+j; };

    void setPixel(int i, int j, int val) { arr[i][j] = val; };
    void setRank(int rank, int val) { arr[0][rank] = val; };

    int  getPixel(int i, int j) { return arr[i][j]; };
    int  getRank(int rank) { return arr[0][rank]; };

  private:
    int    wdth, hght;
    int  **arr;
};

#endif

