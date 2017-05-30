# - Vanilla algorithm

# SIMD
# - Convert into implicit SIMD
# - Add #pragma omp simd (OpenMP)
# - Add #pragma simd  (Cilkplus)
# - Explicit SIMD with AVX immintrin.h

# THREAD
# - OpenMP parallel main loop
# - Cilkplus parallel main loop
# - C++11 thread main loop

# OFFLOADING
# - Whole-program offloading with OpenMP
# - Whole-program offloading with OpenACC
# - Offloading using NVidia CUDA

CC=g++
CXX=g++
CPPFLAGS = -Wall -Wextra
CXXFLAGS = -std=c++14 -Ofast -march=native
LDFLAGS  = -pthread $(shell pkg-config sdl --libs --cflags)

BINARIES = \
	mandelbrot-vanilla \
	\
	mandelbrot-implicit-simd \
	mandelbrot-openmp-simd \
	mandelbrot-cilkplus-simd \
	mandelbrot-explicit-simd \
	\
	mandelbrot-openmp-loop \
	mandelbrot-cilkplus-loop \
	mandelbrot-thread-loop \
	\
	mandelbrot-cuda-loop
	

all: $(BINARIES)

clean:
	rm -f $(BINARIES)
	
$(filter mandelbrot-openmp%,$(BINARIES)): CXXFLAGS += -fopenmp
$(filter mandelbrot-cilk%,$(BINARIES)):   CXXFLAGS += -fcilkplus
$(filter mandelbrot-cilk%,$(BINARIES)):   LDFLAGS  += -lcilkrts
$(filter %explicit-simd,$(BINARIES)):     CXXFLAGS += -msse -msse2 -mssse3 -msse4 -mfma -mavx -mavx2

$(filter mandelbrot-cuda%,$(BINARIES)):   CXX = nvcc -x cu
$(filter mandelbrot-cuda%,$(BINARIES)):   CXXFLAGS = -std=c++11 -O3
$(filter mandelbrot-cuda%,$(BINARIES)):   CPPFLAGS = -Xcompiler '-Wall -Wextra -Ofast'
$(filter mandelbrot-cuda%,$(BINARIES)):   LDFLAGS  = $(shell pkg-config sdl --libs --cflags)
