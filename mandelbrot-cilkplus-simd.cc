#include "common.inc"

/* Vectorized log2() */
template<std::size_t N>
std::array<double,N> plog2(const std::array<double, N>& value)
{
    constexpr int mantissa_bits = 52, exponent_bias = 1022;
    const double  half         = 0.5;
    std::uint64_t half_bits    = reinterpret_cast<const std::uint64_t&>(half);
    // x = frexp(x, &e);
    std::array<int,N> e, lt;
    std::array<uint64_t,N> m;
    std::array<double,N> x, dbl_e, z, y, u, t, result;
    #pragma simd
    for(unsigned n=0; n<N; ++n) { m[n] = reinterpret_cast<const std::uint64_t&>(value[n]); }
    #pragma simd
    for(unsigned n=0; n<N; ++n) { e[n] = m[n] >> mantissa_bits;
                                  m[n] &= std::uint64_t((1ull << mantissa_bits)-1);
                                  m[n] |= half_bits; }
    #pragma simd
    for(unsigned n=0; n<N; ++n) { x[n] = reinterpret_cast<const double&>(m[n]); }
    #pragma simd
    for(unsigned n=0; n<N; ++n) { lt[n] = (x[n] < 1/std::sqrt(2.)) ? -1 : 0;
                                  dbl_e[n] = e[n] + lt[n] - exponent_bias;
                                  z[n] = x[n] - (half + (lt[n] ? 0. : half));
                                  y[n] = half * (x[n] - (lt[n] ? half : 0.)) + half;
                                  x[n] = z[n]/y[n];
                                  z[n] = x[n]*x[n];
                                  u[n] = z[n]      + -3.56722798512324312549E1;
                                  t[n] =             -7.89580278884799154124E-1;
                                  u[n] = u[n]*z[n] +  3.12093766372244180303E2;
                                  t[n] = t[n]*z[n] +  1.63866645699558079767E1;
                                  u[n] = u[n]*z[n] + -7.69691943550460008604E2;
                                  t[n] = t[n]*z[n] + -6.41409952958715622951E1;
                                  y[n] = z[n]* (t[n]/u[n]) + (half+half);
                                  result[n] = x[n]*(y[n]*std::log2(std::exp(1.))) + dbl_e[n]; }
    return result;
}

template<bool WithMoment, unsigned N>
std::array<double,N> Iterate(const double* zr, const std::array<double,N>& zi)
{
    const double escape_radius_squared = ESCAPE_RADIUS_SQUARED;
    const int maxiter = MAXITER;

    std::array<double,N> cr, ci=zi;

    #pragma simd
    for(unsigned n=0; n<N; ++n) { cr[n] = zr[n]; }

    std::array<double,N> sr = cr, si = ci;
    std::array<double,N> dist{};
    std::array<int,N> notescaped, iter, notmoment;

    std::array<double,N> r2,i2,ri;

    #pragma simd
    for(unsigned n=0; n<N; ++n) {
        r2[n] = cr[n] * cr[n];
        i2[n] = ci[n] * ci[n];
        notescaped[n] = ((cr[n]*(1+cr[n]*(8*r2[n]+(16*i2[n]-3)))+i2[n]*(8*i2[n]-3) >= 3./32) & (((cr[n]+1)*(cr[n]+1)+i2[n]) >= 1./16)) ? -1 : 0;
        iter[n] = maxiter & notescaped[n];
    }

    for(;;)
    {
        int n_escaped = 0;
        for(unsigned n=0; n<N; ++n) n_escaped += notescaped[n];
        if(!n_escaped) break;

        #pragma simd
        for(unsigned n=0; n<N; ++n) { dist[n] = notescaped[n] ? (r2[n] + i2[n]) : dist[n];
                                      notescaped[n] &= ((iter[n] != 0) & (dist[n] < escape_radius_squared)) ? -1 : 0;
                                      iter[n] += notescaped[n];
                                      ri[n] = cr[n] * ci[n];
                                      ci[n] = zi[n] + (ri[n] * 2);
                                      cr[n] = zr[n] + (r2[n] - i2[n]);
                                      if(WithMoment)
                                      {
                                          notmoment[n] = iter[n] & (iter[n]-1);
                                          iter[n] &= ((cr[n] != sr[n]) | (ci[n] != si[n])) ? -1 : 0;
                                          sr[n] = notmoment[n] ? sr[n] : cr[n];
                                          si[n] = notmoment[n] ? si[n] : ci[n];
                                      }
                                      r2[n] = cr[n] * cr[n];
                                      i2[n] = ci[n] * ci[n];
                                    }
    }
    std::array<double,N> result;
    result = plog2(dist);
    #pragma simd
    for(unsigned n=0; n<N; ++n) result[n] /= 2;
    result = plog2(result);
    #pragma simd
    for(unsigned n=0; n<N; ++n) result[n] = maxiter - iter[n] + 1 - result[n];
    result = plog2(result);
    #pragma simd
    for(unsigned n=0; n<N; ++n) result[n] = iter[n] ? (result[n] * (4/std::log2(std::exp(1.)))) : 0;
    return result;
}

int main()
{
    bool NeedMoment = true;

    MAINLOOP_START();
    while(MAINLOOP_GET_CONDITION())
    {
        std::vector<unsigned> pixels (Xres * Yres);

        double zr, zi, xscale, yscale; MAINLOOP_SET_COORDINATES();

        constexpr unsigned N = 8;

        unsigned n_inside = 0;

        for(unsigned y=0; y<Yres; ++y)
        {
            std::array<double,Xres> r;
            std::array<double,N>    i;
            std::array<double,Xres> results;
            #pragma simd
            for(unsigned n=0; n<N; ++n)    { i[n] = zi+yscale*int(y-Yres/2); }
            #pragma simd
            for(unsigned x=0; x<Xres; ++x) { r[x] = zr+xscale*int(x-Xres/2); }

            if(NeedMoment)
                for(unsigned x=0; x<Xres/N*N; x += N)
                {
                    auto res = Iterate<true,N>(&r[x], i);
                    #pragma simd
                    for(unsigned n=0; n<N; ++n) results[x+n] = res[n];
                }
            else
                for(unsigned x=0; x<Xres/N*N; x += N)
                {
                    auto res = Iterate<false,N>(&r[x], i);
                    #pragma simd
                    for(unsigned n=0; n<N; ++n) results[x+n] = res[n];
                }

            for(unsigned x=0; x<Xres; ++x) { n_inside += (results[x] == 0.); }
            #pragma simd
            for(unsigned x=0; x<Xres; ++x) { pixels[y*Xres + x] = Color(x,y, results[x]); }
        }

        NeedMoment = n_inside >= (Xres*Yres)/1024;

        MAINLOOP_PUT_RESULT(pixels);
    }
    MAINLOOP_FINISH();
}
