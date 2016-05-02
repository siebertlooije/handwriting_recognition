#ifndef ARR1D_H
#define ARR1D_H

class Array1D {
  public:
    // Constructors and destructor
    Array1D();
    Array1D(int length);
    Array1D(const Array1D &other);
    ~Array1D();
    
    // operators
    const int &operator[](int index) const { return arr[index]; };
    int &operator[](int index) { return arr[index]; };
    
    // misc. methods
    int  length() { return len; };
    void shift(int dx) { x0 += dx; arr += dx;};
    void resetShift() {arr -= x0; x0 = 0;};
    void copy(const Array1D &other);
    void setSize(int length);
    void setValue(int value);
    void deallocate();
  private:
    int    len;
    int    x0;
    int   *arr;
};

#endif

