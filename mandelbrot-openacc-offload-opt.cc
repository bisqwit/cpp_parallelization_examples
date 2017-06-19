#include "common.inc"

#pragma acc routine
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
    return iter ? std::log2( maxiter-iter + 1 - std::log2(std::log2(dist) / 2)) * (4/std::log2(std::exp(1.))) : 0;
}

int main(int argc, char **argv)
{
    static double results[Xres*Yres];

    bool NeedMoment = true;

    std::vector<unsigned> pixels (Xres * Yres);

    MAINLOOP_START(1);
    while(MAINLOOP_GET_CONDITION())
    {

        double zr, zi, xscale, yscale; MAINLOOP_SET_COORDINATES();

        if(NeedMoment)
        {
            #pragma acc parallel loop gang worker vector copyout(results[0:Xres*Yres]) collapse(2)
            for(unsigned y=0; y<Yres; ++y)
                for(unsigned x=0; x<Xres; ++x)
                    results[y*Xres+x] = Iterate<true>( zr+xscale*int(x-Xres/2), zi+yscale*int(y-Yres/2) );
        }
        else
        {
            #pragma acc parallel loop gang worker vector copyout(results[0:Xres*Yres]) collapse(2)
            for(unsigned y=0; y<Yres; ++y)
                for(unsigned x=0; x<Xres; ++x)
                    results[y*Xres+x] = Iterate<false>( zr+xscale*int(x-Xres/2), zi+yscale*int(y-Yres/2) );
        }

        unsigned n_inside = std::count_if(results, results+Xres*Yres, std::bind1st(std::equal_to<double>(), 0.));

        NeedMoment = n_inside >= (Xres*Yres)/1024;

        for(unsigned y=0; y<Yres; ++y)
            for(unsigned x=0; x<Xres; ++x)
                pixels[y*Xres + x] = Color(x,y, results[y*Xres+x]);

        MAINLOOP_PUT_RESULT(pixels);
    }
    MAINLOOP_FINISH();
}
