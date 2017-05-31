DEFS=-DMAXITER=6000 -DESCAPE_RADIUS_SQUARED=6*6 -DESCAPE_RADIUS_MANHATTAN=6

# At 6000 iterations, escape 6*6, res 212x120
#  fma:        209
#  avx2:       200
#  avx:        106
#  sse4:       71
#  sse2:       73
#
#  implicit:   148
#  openmpsimd: 142
#  cilksimd:   142
#  vanilla:    117

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
CPPFLAGS = -Wall -Wextra $(DEFS)
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
	mandelbrot-openmp-offload \
	mandelbrot-openacc-offload \
	mandelbrot-cuda-offload
	

all: $(BINARIES)

clean:
	rm -f $(BINARIES)
	
$(filter mandelbrot-openmp%,$(BINARIES)): CXXFLAGS += -fopenmp
$(filter mandelbrot-cilk%,$(BINARIES)):   CXXFLAGS += -fcilkplus
$(filter mandelbrot-cilk%,$(BINARIES)):   LDFLAGS  += -lcilkrts
$(filter %explicit-simd,$(BINARIES)):     CXXFLAGS += -march=native $(PLATFORM_OPTS)
$(filter mandelbrot-openacc%,$(BINARIES)): CXXFLAGS += -fopenacc

$(filter mandelbrot-openmp-offload,$(BINARIES)):  CXXFLAGS += -foffload=x86_64-intelmicemul-linux-gnu -foffload=nvptx-none
$(filter mandelbrot-openacc-offload,$(BINARIES)): CXXFLAGS += -foffload=x86_64-intelmicemul-linux-gnu -foffload=nvptx-none

$(filter mandelbrot-cuda%,$(BINARIES)):   CXX = nvcc -x cu
$(filter mandelbrot-cuda%,$(BINARIES)):   CXXFLAGS = -std=c++11 -O3
$(filter mandelbrot-cuda%,$(BINARIES)):   CPPFLAGS = -Xcompiler '-Wall -Wextra -Ofast' $(DEFS)
$(filter mandelbrot-cuda%,$(BINARIES)):   LDFLAGS  = $(shell pkg-config sdl --libs --cflags)
