# Compiler
CC = clang++

# Directories
ROOTDIR = ../..
# Bin directory
BINDIR = ${ROOTDIR}/bin/

# DISP Framework src dir
DISPDIR = ${DISP_HOME}/src/core

# Flags
CFLAGS = -std=c++11 -O3 -Wall -Wextra 
CFLAGS += -g -fPIC

# Common data structures and utilities classes
COMMONDIR = ${ROOTDIR}/src/common

HDF5LIB = ${HDF5_HOME}/lib
HDF5INC = ${HDF5_HOME}/include

MPILIB = ${MPI_HOME}/lib
MPIINC = ${MPI_HOME}/inc

LIBS = -lm -lhdf5 -lz -lmpi 
LIBDIRS = -L$(HDF5LIB) -L$(MPILIB)
INCLUDES = -I$(DISPDIR) -I${COMMONDIR} -I$(HDF5INC) -I$(MPIINC)

# Executable/reconstruction objects
SIRT_OBJS = trace_io_sim_main.o
COMMON_OBJS = trace_h5io.o

# Executables
PROGS = trace_io_sim

.PHONY: default clean

all: $(PROGS)

# SIRT
trace_io_sim: $(COMMON_OBJS) $(SIRT_OBJS)
	$(CC) $(CFLAGS) -o $@ $(SIRT_OBJS) $(COMMON_OBJS) $(LIBDIRS) $(LIBS) 

trace_io_sim_main.o: trace_io_sim_main.cc
	$(CC) $(CFLAGS) -c trace_io_sim_main.cc $(INCLUDES)

trace_h5io.o: $(COMMONDIR)/trace_h5io.cc $(COMMONDIR)/trace_h5io.h
	$(CC) $(CFLAGS) -c $(COMMONDIR)/trace_h5io.cc $(INCLUDES)

trace_utils.o: $(COMMONDIR)/trace_utils.cc $(COMMONDIR)/trace_utils.h
	$(CC) $(CFLAGS) -c $(COMMONDIR)/trace_utils.cc $(INCLUDES)

install:
	mv $(PROGS) $(BINDIR)

clean:
	rm -f $(PROGS) *.o *.a *~ *.lst *.tmp .pure *.bak *.log
