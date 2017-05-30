#include "common.inc"

template<unsigned N>
std::array<double,N> Iterate(std::array<double,N> zr, std::array<double,N> zi)
{
    const double escape_radius_squared = 6*6;
    const int maxiter = 8100;
    std::array<double,N> cr = zr, sr = cr;
    std::array<double,N> ci = zi, si = ci;
    std::array<double,N> dist{};
    std::array<int,N> escaped{}, iter, moment;

    for(unsigned n=0; n<N; ++n) iter[n] = maxiter;
    for(;;)
    {
        int n_escaped = 0;
        #pragma simd
        for(unsigned n=0; n<N; ++n) n_escaped += escaped[n];
        if(n_escaped >= int(N)) break;

        std::array<double,N> r2,i2,ri;
        #pragma simd
        for(unsigned n=0; n<N; ++n) { r2[n] = cr[n] * cr[n];
                                      i2[n] = ci[n] * ci[n];
                                      if(!escaped[n]) { dist[n] = r2[n] + i2[n]; }
                                      escaped[n] = escaped[n] || !iter[n] || (dist[n] >= escape_radius_squared);
                                      iter[n] -= !escaped[n];
                                      ri[n] = cr[n] * ci[n];
                                      ci[n] = zi[n] + (ri[n] * 2);
                                      cr[n] = zr[n] + (r2[n] - i2[n]);
                                      moment[n] = iter[n] & (iter[n]-1);
                                      if(cr[n] == sr[n] && ci[n] == si[n]) { iter[n] = 0; }
                                      if(!moment[n]) { sr[n] = cr[n]; si[n] = ci[n]; } }
    }
    std::array<double,N> result;
    for(unsigned n=0; n<N; ++n) result[n] = iter[n] ? std::log( maxiter-iter[n] + 1 - std::log2(std::log2(dist[n]) / 2)) * 4 : 0;
    return result;
}

int main()
{
    while(GetTime() < 5)
    {
        std::vector<unsigned> pixels (Xres * Yres);

        double zr = -0.743639266077433, zi = 0.131824786875559, scale = 4. * std::pow(2, -std::min(GetTime(),53.)*0.7);
        double xscale = scale/Yres, yscale = -scale/Yres;
        constexpr unsigned N=8;

        for(unsigned y=0; y<Yres; ++y)
        {
            std::array<double,N> i;
            #pragma simd
            for(unsigned n=0; n<N; ++n) { i[n] = zi+yscale*int(y-Yres/2); }

            for(unsigned x=0; x<Xres/N*N; x += N)
            {
                std::array<double,N> r;
                #pragma simd
                for(unsigned n=0; n<N; ++n) { r[n] = zr+xscale*int(x+n-Xres/2); }

                auto results = Iterate<N>(r,i);
                for(unsigned n=0; n<N; ++n) { pixels[y*Xres + x+n] = Color(x+n,y, results[n]); }
            }
        }

        display.Put(pixels);
    }
    std::printf("\n%u frames rendered\n", display.frame);
}
