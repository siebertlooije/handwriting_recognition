#include <iostream>

class X {
 public:
  X() { std:: cout << "constructor X\n"; }
};

class Y {
 public:
  Y() { std:: cout << "constructor Y\n"; 
        p = new X [10];
      }
  private :
    X *p; 
};

int main (int argc, char **argv) {
  Y y;
  return 0;
}
