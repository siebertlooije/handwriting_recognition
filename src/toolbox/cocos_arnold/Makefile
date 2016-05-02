CXXFLAGS=-Wall -pedantic -O3 

OBJECTS= arr1d.o arr2d.o conncomp.o error.o \
         imageio.o threshold.o \
         main.o

all: ${OBJECTS}
	g++ ${CXXFLAGS} -o main ${OBJECTS}
clean:
	rm -f *~
	rm -f *.o
	rm -f main
