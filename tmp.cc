#include <cmath>
const unsigned N = 64;

//struct kuru
//{
int x[N] __attribute__((aligned(32)));
long long y[N] __attribute__((aligned(32)));
long long z[N] __attribute__((aligned(32)));

double a[N] __attribute__((aligned(32)));
double b[N] __attribute__((aligned(32)));
double c[N] __attribute__((aligned(32)));
//} test __attribute__((aligned(128)));

void testaa()
{
    #pragma omp simd
    for(unsigned n=0; n<N; ++n) a[n] = std::sqrt(double(y[n]));
}


