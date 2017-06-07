#include "common.inc"

#pragma omp declare target

double mylog2(double value)
{
    constexpr int mantissa_bits = 52, exponent_bias = 1022;
    const double  half         = 0.5;
    std::uint64_t half_bits    = reinterpret_cast<const std::uint64_t&>(half);
    int e,lt;
    uint64_t m;
    double x, dbl_e, z, y, u, t;
    m = reinterpret_cast<const std::uint64_t&>(value);
    e = m >> mantissa_bits; // frexp(). e = exponent, m = mantissa
    m &= std::uint64_t((1ull << mantissa_bits)-1);
    m |= half_bits;
    x = reinterpret_cast<const double&>(m);
    lt = (x < 1/std::sqrt(2.)) ? -1 : 0;
    dbl_e = e + lt - exponent_bias;
    z = x - (half + (lt ? 0. : half));
    y = half * (x - (lt ? half : 0.)) + half;
    x = z/y;
    z = x*x;
    u = z   + -3.56722798512324312549E1;
    t =       -7.89580278884799154124E-1;
    u = u*z +  3.12093766372244180303E2;
    t = t*z +  1.63866645699558079767E1;
    u = u*z + -7.69691943550460008604E2;
    t = t*z + -6.41409952958715622951E1;
    y = z* (t/u) + (half+half);
    return x*(y*std::log2(std::exp(1.))) + dbl_e;
}

template<bool WithMoment>
double Iterate(double zr, double zi)
{
    const double escape_radius_squared = ESCAPE_RADIUS_SQUARED;
    const int maxiter = MAXITER;
    double cr = zr, sr = cr;
    double ci = zi, si = ci;
    double dist;
    int iter = maxiter, notescaped = -1;

    if(zr*(1+zr*(8*zr*zr+(16*zi*zi-3)))+zi*zi*(8*zi*zi-3) < 3./32 || ((zr+1)*(zr+1)+zi*zi)<1./16) { iter=0; }

    while(notescaped)
    {
        double r2 = cr * cr;
        double i2 = ci * ci;
        dist = r2 + i2;

        notescaped &= ((iter != 0) & (dist < escape_radius_squared)) ? -1 : 0;
        iter += notescaped;

        double ri = cr * ci;
        ci = zi + (ri * 2);
        cr = zr + (r2 - i2);

        if(WithMoment)
        {
            bool notmoment = iter & (iter-1);
            iter = (cr == sr && ci == si) ? 0 : iter;
            sr = notmoment ? sr : cr;
            si = notmoment ? si : ci;
        }
    }
    return iter ? mylog2( maxiter-iter + 1 - mylog2(mylog2(dist) / 2)) * (4/std::log2(std::exp(1.))) : 0;
}
#pragma omp end declare target

int main()
{
    static double results[Xres*Yres];

    bool NeedMoment = true;

    MAINLOOP_START(1);
    while(MAINLOOP_GET_CONDITION())
    {
        std::vector<unsigned> pixels (Xres * Yres);

        double zr, zi, xscale, yscale; MAINLOOP_SET_COORDINATES();

        if(NeedMoment)
        {
            #pragma omp target teams distribute parallel for collapse(2) map(to:zr,zi,xscale,yscale) map(from:results[0:Xres*Yres])
            for(unsigned y=0; y<Yres; ++y)
                for(unsigned x=0; x<Xres; ++x)
                    results[y*Xres+x] = Iterate<true>( zr+xscale*int(x-Xres/2), zi+yscale*int(y-Yres/2) );
        }
        else
        {
            #pragma omp target teams distribute parallel for collapse(2) map(to:zr,zi,xscale,yscale) map(from:results[0:Xres*Yres])
            for(unsigned y=0; y<Yres; ++y)
                for(unsigned x=0; x<Xres; ++x)
                    results[y*Xres+x] = Iterate<false>( zr+xscale*int(x-Xres/2), zi+yscale*int(y-Yres/2) );
        }

        unsigned n_inside = std::count_if(results, results+Xres*Yres, std::bind1st(std::equal_to<double>(), 0.));

        NeedMoment = n_inside >= (Xres*Yres)/1024;

        #pragma omp parallel for /* This part is run natively */
        for(unsigned y=0; y<Yres; ++y)
            for(unsigned x=0; x<Xres; ++x)
                pixels[y*Xres + x] = Color(x,y, results[y*Xres+x]);

        MAINLOOP_PUT_RESULT(pixels);
    }
    MAINLOOP_FINISH();
}
