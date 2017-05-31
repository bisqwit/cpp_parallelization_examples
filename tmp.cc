#include <cmath>
const unsigned N = 8;

//struct kuru
//{
int x[N] __attribute__((aligned(32)));
int y[N] __attribute__((aligned(32)));

//} test __attribute__((aligned(128)));

void testaa()
{
    #pragma omp simd
    for(unsigned n=0; n<N; ++n) x[n] = x[n] == y[n] ? -1 : 0;
}


