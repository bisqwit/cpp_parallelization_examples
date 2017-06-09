DEFS=-DMAXITER=8000 -DESCAPE_RADIUS_SQUARED=6*6

# PART 1/3: SIMD
# - Convert into implicit SIMD
# - Add #pragma omp simd (OpenMP)
# - Add #pragma simd  (Cilkplus)
# - Explicit SIMD with AVX immintrin.h

CC=g++
CXX=g++
CPPFLAGS = -Wall -Wextra $(DEFS) -Wno-clobbered
CXXFLAGS = -std=c++14 -Ofast -march=native
LDFLAGS  = -pthread $(shell pkg-config sdl --libs)
CPPFLAGS +=         $(shell pkg-config sdl --cflags --libs)

BINARIES = \
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

run: $(BINARIES)
	for s in $^ ; do ./$$s ; done

$(filter mandelbrot-openmp%,$(BINARIES)): CXXFLAGS += -fopenmp
$(filter mandelbrot-cilk%,$(BINARIES)):   CXXFLAGS += -fcilkplus
$(filter mandelbrot-cilk%,$(BINARIES)):   LDFLAGS  += -lcilkrts
$(filter %explicit-simd,$(BINARIES)):     CXXFLAGS += -march=native $(PLATFORM_OPTS)

$(BINARIES): CPPFLAGS += -DPROG_NAME="\"$(subst mandelbrot-,,$(subst .o,,$@))\""
