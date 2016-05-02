#include <fstream>
#include <string.h>

#include "arr2d.h"
#include "error.h"

void readImage (char *fileName, Array2D &im) {
  int c, w, h;
  std::ifstream in(fileName);

  if (in==0) {
    error ("readImage:", "could not open file!\n");
  }
  // Parse header
  c = in.get();
  if (c!='P') {
    error ("readImage:", "file is not a PGM file!\n");
  }
  c = in.get();
  if (c!='5') {
    error ("readImage:", "only PGM type P5 is supported!\n");
  }
  while ((c=in.get()) != '\n');
  
  c = in.get();
  while (c=='#') { // skip commentline
    while ((c=in.get()) != '\n');
    c = in.get();
  }
  in.unget();

  // read width, height of image and allocate memory
  in >> w >> h;
  while ((c=in.get()) != '\n');
  im.reSize(w, h);

  // read dummy value of 255
  while ((c=in.get()) != '\n');  
  // read image data
  unsigned char *buf = new unsigned char [w];
  for (int i=0; i<h; ++i) {
    in.read((char*)buf, w);
    for (int j=0; j<w; ++j) {
      im[i][j] = buf[j];
    }
  }
  delete [] buf;
  in.close();
}

void writeImage (char *fileName, Array2D &im) {
  std::ofstream out(fileName);
  int w, h;

  w = im.width();
  h = im.height();

  // write header
  out << "P5\n";
  out << w << " " << h << "\n";
  out << "255\n";

  unsigned char *buf = new unsigned char [w];
  for (int i=0; i<h; ++i) {
    for (int j=0; j<w; ++j) {
      buf[j] = im[i][j];
    }
    out.write((char*)buf, w);
  }
  delete [] buf;
  out.close();
}

