CC=gcc
CPP=c++
PYINCLUDE=-I/usr/include/python2.7 -I/usr/include/python2.6 -I/usr/include/python2.5 -I/usr/include/python2.4
OFLAGS=-Icocos_arnold -c -Wall -O4 -fPIC
SOFLAGS=-shared -Wall -O4 -fPIC $(PYINCLUDE)

.PHONY: all
all: pamImage.py cocoslib.py croplib.py
	@rm *.o

%.o: %.cpp
	$(CPP) $(OFLAGS) -pedantic $<

pamImage.py: pamImage.o pamImage.i
	swig -c++ -python pamImage.i
	$(CPP) $(SOFLAGS) -fPIC -shared pamImage.cpp pamImage_wrap.cxx -o _pamImage.so
	rm -f pamImage_wrap.cxx

cocoslib.py: pamImage.o cocoslib.i cocoslib.o cocos_arnold/arr2d.o cocos_arnold/conncomp.o
	swig -c++ -python cocoslib.i
	$(CPP) $(SOFLAGS) -Icocos_arnold cocoslib.o cocoslib_wrap.cxx arr2d.o conncomp.o pamImage.o -o _cocoslib.so
	rm -f cocoslib_wrap.cxx

croplib.py: pamImage.o croplib.i croplib.o
	swig -c++ -python croplib.i
	$(CPP) $(SOFLAGS) -shared croplib.o croplib_wrap.cxx pamImage.o -o _croplib.so
	rm -f croplib_wrap.cxx

clean:
	@rm -f *.o *.so pamImage.py cocoslib.py croplib.py
	@rm -f *.pyc # Not created by this Makefile
