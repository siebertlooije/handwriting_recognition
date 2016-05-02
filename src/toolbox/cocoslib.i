%module cocoslib
%include "typemaps.i"
%apply long *OUTPUT { long &left, long &right, long &top, long &bottom };

%{
#include "pamImage.h"
#include "cocoslib.h"
#include "ailib.h"
%}

%include "cocoslib.h"
