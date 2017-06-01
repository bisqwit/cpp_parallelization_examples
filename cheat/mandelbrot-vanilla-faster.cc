#include "common.inc"

double Iterate(double zr, double zi)
{
    const double escape_radius_manhattan = ESCAPE_RADIUS_MANHATTAN;
    const int maxiter = MAXITER;
    double cr = zr, sr = cr;
    double ci = zi, si = ci;
    double dist = 0;
    int iter    = maxiter;

    if(zr*(1+zr*(8*zr*zr+(16*zi*zi-3)))+zi*zi*(8*zi*zi-3) < 3./32 || ((zr+1)*(zr+1)+zi*zi)<1./16) { act=false; iter=0; }

    while(iter)
    {
        if(std::abs(cr) + std::abs(ci) >= escape_radius_manhattan)
        {
            dist = cr*cr + ci*ci;
            break;
        }

        --iter;
        double reim2 = ci * cr * 2;
        cr = (cr - ci) * (cr + ci) + zr;
        ci = reim2 + zi;

        if(iter&(iter-1)) { if(cr == sr && ci == si) { iter=0; }}
        else              { sr = cr; si = ci; }
    }
    return iter ? std::log( maxiter-iter + 1 - std::log2(std::log2(dist) / 2)) * 4 : 0;
}

int main()
{
    while(GetTime() < 5)
    {
        std::vector<unsigned> pixels (Xres * Yres);

        double zr = -0.743639266077433, zi = 0.131824786875559, scale = 4. * std::pow(2, -std::min(GetTime(),53.)*0.7);
        double xscale = scale/Yres, yscale = -scale/Yres;

        for(unsigned y=0; y<Yres; ++y)
            for(unsigned x=0; x<Xres; ++x)
                pixels[y*Xres + x] = Color(x,y,Iterate( zr+xscale*int(x-Xres/2), zi+yscale*int(y-Yres/2) ));

        display.Put(pixels);
    }
    std::printf("\n%u frames rendered\n", display.frame);
}
