CC = CC

CFLAGS = -g -O2 -I../include/Cubism/ -fopenmp -Wall

.PHONY: all clean

all: main.cpp
	$(CC) $(CFLAGS) -o main  main.cpp

clean:
	rm -f main *.o
