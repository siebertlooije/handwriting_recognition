#include <iostream>
#include <stdlib.h>

void error (char *funcname, char *errmsg) {
  std::cerr << funcname << ": " << errmsg << "\n";
  exit(EXIT_FAILURE);
}
