DEFS=-DMAXITER=8000 -DESCAPE_RADIUS_SQUARED=6*6
#DEFS=-DMAXITER=6000 -DESCAPE_RADIUS_SQUARED=6*6
#-DESCAPE_RADIUS_MANHATTAN=6

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

# - Offloading using NVidia CUDA, two streams & uint16
# - Offloading using NVidia CUDA, two streams & uint16 _and_ explicit SIMD _and_ C++11 threads

CC=g++
CXX=g++
CPPFLAGS = -Wall -Wextra $(DEFS) -Wno-clobbered
CXXFLAGS = -std=c++14 -Ofast -march=native
LDFLAGS  = -pthread $(shell pkg-config sdl --libs)
CPPFLAGS +=         $(shell pkg-config sdl --cflags --libs)

BINARIES = \
	mandelbrot-cuda-offload3 \
	mandelbrot-cuda-offload2 \
	mandelbrot-cuda-offload \
	mandelbrot-openmp-offload \
	mandelbrot-openacc-offload \
	\
	mandelbrot-openmp-loop \
	mandelbrot-cilkplus-loop \
	mandelbrot-thread-loop \
	\
	mandelbrot-implicit-simd \
	mandelbrot-openmp-simd \
	mandelbrot-cilkplus-simd \
	mandelbrot-explicit-simd \
	\
	mandelbrot-vanilla \
	\
	

all: $(BINARIES)

clean:
	rm -f $(BINARIES)

#run:		mandelbrot-cuda-offload2 mandelbrot-cuda-offload mandelbrot-cilkplus-loop mandelbrot-thread-loop \
#		mandelbrot-implicit-simd mandelbrot-openmp-simd mandelbrot-cilkplus-simd mandelbrot-explicit-simd \
#		mandelbrot-vanilla mandelbrot-openmp-loop
run: $(BINARIES)
	for s in $^ ; do ./$$s ; done

$(filter mandelbrot-openmp%,$(BINARIES)): CXXFLAGS += -fopenmp
$(filter mandelbrot-cilk%,$(BINARIES)):   CXXFLAGS += -fcilkplus
$(filter mandelbrot-cilk%,$(BINARIES)):   LDFLAGS  += -lcilkrts
$(filter %explicit-simd,$(BINARIES)):     CXXFLAGS += -march=native $(PLATFORM_OPTS)
$(filter mandelbrot-openacc%,$(BINARIES)): CXXFLAGS += -fopenacc

$(filter mandelbrot-openmp-offload,$(BINARIES)):  CXXFLAGS += -foffload=x86_64-intelmicemul-linux-gnu -foffload=nvptx-none
$(filter mandelbrot-openacc-offload,$(BINARIES)): CXXFLAGS += -foffload=x86_64-intelmicemul-linux-gnu -foffload=nvptx-none

$(filter mandelbrot-cuda%,$(BINARIES)):   CXX = nvcc -x cu
$(filter mandelbrot-cuda%,$(BINARIES)):   CC  = nvcc -Xcompiler -fopenmp
$(filter mandelbrot-cuda%,$(BINARIES)):   CXXFLAGS = -std=c++11 -O3 $(shell pkg-config sdl --cflags)
$(filter mandelbrot-cuda%,$(BINARIES)):   CPPFLAGS = -Xcompiler '-Wall -Wextra -Ofast -fopenmp' $(DEFS)
$(filter mandelbrot-cuda%,$(BINARIES)):   LDFLAGS  = $(shell pkg-config sdl --libs --cflags)

#$(BINARIES): common.inc
#$(filter mandelbrot-implicit-simd%,$(BINARIES)):   CXX = clang++-5.0

$(BINARIES): CPPFLAGS += -DPROG_NAME="\"$(subst mandelbrot-,,$(subst .o,,$@))\""

mandelbrot-cuda-offload3:     CXXFLAGS += -Xcompiler '-march=native $(PLATFORM_OPTS)'
mandelbrot-cuda-offload3: mandelbrot-cuda-offload3.o mandelbrot-explicit-simd-helper.o mandelbrot-thread-loop-helper.o

mandelbrot-explicit-simd-helper.o: CXX = g++ -fopenmp
mandelbrot-explicit-simd-helper.o: CPPFLAGS = -Wall -Wextra $(DEFS) -Wno-clobbered
mandelbrot-explicit-simd-helper.o: CXXFLAGS = -std=c++14 -Ofast -march=native

mandelbrot-thread-loop-helper.o: CXX = g++ -fopenmp
mandelbrot-thread-loop-helper.o: CPPFLAGS = -Wall -Wextra $(DEFS) -Wno-clobbered
mandelbrot-thread-loop-helper.o: CXXFLAGS = -std=c++14 -Ofast -march=native
