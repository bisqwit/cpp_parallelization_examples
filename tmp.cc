#include <cmath>
const unsigned N = 8;

//struct kuru
//{
long x[N] __attribute__((aligned(32)));

//} test __attribute__((aligned(128)));

int  testaa()
{
    register long y[N];
    for(unsigned n=0; n<N; ++n) y[n] = x[n];

    for(unsigned n=0; n<N/2; ++n) y[n] += y[N/2+n]; // 0,1,2,3 += 4,5,6,7
    for(unsigned n=0; n<N/4; ++n) y[n] += y[N/4+n]; // 0,1 += 2,3
    for(unsigned n=0; n<N/8; ++n) y[n] += y[N/8+n]; // 0,1 += 2,3

    return y[0];
}


