SHELL := /bin/bash

CC = mpic++

precision ?= single
hdf5precision ?= single
nonuniform ?= no

extra ?=

CFLAGS = -fopenmp -g -O2 -I../../include
CFLAGS+= $(extra)
# CFLAGS += -I${HDF5_ROOT}/include
CFLAGS+= -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align
CFLAGS+= -Wwrite-strings -Wmissing-declarations -Wredundant-decls
# CFLAGS+= -Wlogical-op
CFLAGS+= -Wshadow -Woverloaded-virtual -Wuninitialized
CFLAGS+= -Wpedantic -Wno-unused-parameter # does not make much sense with function overloading


LIBS =
# LIBS += -L${HDF5_ROOT}/lib

.PHONY: all clean

$(warning using precision=$(precision))
$(warning using hdf5precision=$(hdf5precision))
$(warning using nonuniform=$(nonuniform))
ifeq ("$(precision)","double")
	CFLAGS += -D_DOUBLE_
endif

ifeq ("$(hdf5precision)","double")
	CFLAGS += -D_HDF5_DOUBLE_PRECISION_
endif

ifeq ("$(nonuniform)","yes")
	CFLAGS += -D_NONUNIFORM_
endif

all: testDumpsMPI.cpp
	$(CC) $(CFLAGS) -o testDumpsMPI testDumpsMPI.cpp $(LIBS) -lhdf5

clean:
	rm -f testDumpsMPI *.o *.h5 *.xmf *.bin *.zbin
