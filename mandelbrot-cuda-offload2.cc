#include "common.inc"
#include "helper_cuda.h"

template<bool WithMoment>
void __global__ Iterate(double zr, double zi, double xscale, double yscale, std::uint16_t* results)
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

    if(zr*(1+zr*(8*zr*zr+(16*zi*zi-3)))+zi*zi*(8*zi*zi-3) < 3./32 || ((zr+1)*(zr+1)+zi*zi)<1./16) { act=false; iter=0; }

    while(act)
    {
        double r2 = cr * cr;
        double i2 = ci * ci;
        dist = act ? r2 + i2 : dist;
        act = iter && dist < escape_radius_squared ? act : false;
        iter -= act;
        double ri = cr * ci;
        ci = zi + (ri * 2);
        cr = zr + (r2 - i2);

        if(WithMoment)
        {
            bool moment = iter & (iter-1);
            iter = (cr == sr && ci == si) ? 0 : iter;
            sr = moment ? sr : cr;
            si = moment ? si : ci;
        }
    }
    results[slotno] = iter ? int(std::log2( maxiter-iter + 1 - std::log2(std::log2(dist) / 2)) * (8*4/std::log2(std::exp(1.)))) : 0;
}

constexpr unsigned npixels = Xres * Yres, nthreads = 128, nblocks = (npixels + nthreads - 1) / nthreads;
constexpr unsigned num_streams = 2;

int main()
{
    static std::uint16_t results[num_streams][npixels], *pointers[num_streams]{};
    cudaStream_t streams[num_streams];

    unsigned stream_number = 0, streams_busy = 0;
    bool     stream_busy[num_streams];

    for(unsigned n=0; n<num_streams; ++n)
    {
        checkCudaErrors(cudaMalloc((void**)&pointers[n], sizeof(results[0])));
        checkCudaErrors(cudaStreamCreate(&streams[n]));
        stream_busy[n] = false;
    }

    bool NeedMoment = true;

    MAINLOOP_START(1);
    while(MAINLOOP_GET_CONDITION() || streams_busy)
    {
        double zr, zi, xscale, yscale; MAINLOOP_SET_COORDINATES();

        if(stream_busy[stream_number])
        {
            stream_busy[stream_number] = false; --streams_busy;
            checkCudaErrors(cudaStreamSynchronize(streams[stream_number]));

            const std::uint16_t* r = results[stream_number];
            unsigned n_inside = std::count_if(r, r+npixels, std::bind1st(std::equal_to<std::uint16_t>(), 0));
            NeedMoment = n_inside >= (Xres*Yres)/1024;

            std::vector<unsigned> pixels (Xres * Yres);

            #pragma omp parallel for
            for(unsigned y=0; y<Yres; ++y)
                for(unsigned x=0; x<Xres; ++x)
                    pixels[y*Xres + x] = Color(x,y, r[y*Xres+x]/8.);

            MAINLOOP_PUT_RESULT(pixels);
        }

        if(MAINLOOP_GET_CONDITION())
        {
            if(NeedMoment)
                Iterate<true><<<nblocks, nthreads, 0, streams[stream_number]>>>( zr, zi, xscale, yscale, pointers[stream_number]);
            else
                Iterate<false><<<nblocks, nthreads, 0, streams[stream_number]>>>( zr, zi, xscale, yscale, pointers[stream_number]);

            cudaMemcpyAsync(results[stream_number], pointers[stream_number], sizeof(results[0]), cudaMemcpyDeviceToHost, streams[stream_number]);
            stream_busy[stream_number] = true; ++streams_busy;
        }

        stream_number = (stream_number+1)%num_streams;
    }
    MAINLOOP_FINISH();
}
