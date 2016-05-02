#include <fstream>
#include <string.h>
#include <stdlib.h>

#include "ailib.h"

int main (int argc, char *argv[]) {  
  Array2D im;
  int ncc;

  /************** 4-connected ************/
  readImage (argv[1], im);
  ncc = componentLabeling4(im);
  printf ("ncc4=%d\n", ncc);
/*
  readImage (argv[1], im);
  ncc = componentLabeling4(im, 8);
  printf ("*ncc4=%d\n", ncc);

  readImage (argv[1], im);
  int nb4[2][2] = {{-1,0},{0,-1}};
  ncc = componentLabeling(2, nb4, im);
  printf ("**ncc4=%d\n", ncc);

  readImage (argv[1], im);
  ncc = componentLabeling(2, nb4, im, 8);
  printf ("**ncc4=%d\n", ncc);

*/

  /************** 8-connected ************/
/*
  readImage (argv[1], im);
  ncc = componentLabeling8(im);
  printf ("ncc8=%d\n", ncc);

  readImage (argv[1], im);
  ncc = componentLabeling8(im, 8);
  printf ("*ncc8=%d\n", ncc);

  readImage (argv[1], im);
  int nb8[4][2] = {{-1,-1},{-1,0},{-1,1},{0,-1}};
  ncc = componentLabeling(4, nb8, im);
  printf ("**ncc8=%d\n", ncc);

  readImage (argv[1], im);
  ncc = componentLabeling(4, nb8, im, 8);
  printf ("**ncc8=%d\n", ncc);


  writeImage ("out.pgm", im);
*/
  return (EXIT_SUCCESS);
}
