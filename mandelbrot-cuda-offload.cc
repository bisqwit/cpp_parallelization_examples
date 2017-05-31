#include "common.inc"
#include "helper_cuda.h"

void __global__ Iterate(double zr, double zi, double xscale, double yscale, double* results)
{
    const double escape_radius_squared = ESCAPE_RADIUS_SQUARED;
    const int maxiter = MAXITER;

    unsigned slotno = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned x = slotno % Xres, y = slotno / Xres;
    if(y >= Yres) return;

    double cr = zr += xscale*int(x-Xres/2), sr = cr;
    double ci = zi += yscale*int(y-Yres/2), si = ci;
    double dist = 0;
    int iter    = maxiter;
    bool act    = true;

    while(act)
    for(unsigned n=0; n<10; ++n)
    {
        double r2 = cr * cr;
        double i2 = ci * ci;
        dist = act ? r2 + i2 : dist;
        act = iter && dist < escape_radius_squared ? act : false;
        iter -= act;
        double ri = cr * ci;
        ci = zi + (ri * 2);
        cr = zr + (r2 - i2);
        bool moment = iter & (iter-1);
        iter = (cr == sr && ci == si) ? 0 : iter;
        sr = moment ? sr : cr;
        si = moment ? si : ci;
    }
    results[slotno] = iter ? std::log( maxiter-iter + 1 - std::log2(std::log2(dist) / 2)) * 4 : 0;
}

constexpr unsigned npixels = Xres * Yres, nthreads = 128, nblocks = (npixels + nthreads - 1) / nthreads;

int main()
{
    double results[npixels], *p = NULL;
    checkCudaErrors(cudaMalloc((void**)&p, sizeof(results))); assert(p != NULL);

    while(GetTime() < 5)
    {
        std::vector<unsigned> pixels (Xres * Yres);

        double zr = -0.743639266077433, zi = 0.131824786875559, scale = 4. * std::pow(2, -std::min(GetTime(),53.)*0.7);
        double xscale = scale/Yres, yscale = -scale/Yres;

        Iterate<<<nblocks, nthreads, 0>>>( zr, zi, xscale, yscale, p);

        checkCudaErrors(cudaMemcpy(results, p, sizeof(results), cudaMemcpyDeviceToHost));

        for(unsigned y=0; y<Yres; ++y)
            for(unsigned x=0; x<Xres; ++x)
                pixels[y*Xres + x] = Color(x,y, results[y*Xres+x]);

        display.Put(pixels);
    }
    std::printf("\n%u frames rendered\n", display.frame);
}
