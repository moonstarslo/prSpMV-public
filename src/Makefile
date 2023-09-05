.PHONY: all clean

CXX = g++
OBJDUMP = objdump
CFLAGS = -ggdb3 -O2 -Wall -Wextra -std=c++17
CFLAGS += -march=skylake-avx512
LDFLAGS = -lm -static -fopenmp

all: spmv spmv_test

spmv: spmv_csr_normal.cpp 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $< 

spmv_test: spmv_csr_normal_malloc.cpp 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $< 

clean:
	rm spmv spmv_test
