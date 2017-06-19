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
    double dist;
    int iter = maxiter, notescaped = -1;

    if(zr*(1+zr*(8*zr*zr+(16*zi*zi-3)))+zi*zi*(8*zi*zi-3) < 3./32 || ((zr+1)*(zr+1)+zi*zi)<1./16) { notescaped=iter=0; }

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
            if(cr==sr && ci==si) iter=0;
            if(!notmoment) sr = cr;
            if(!notmoment) si = ci;
        }
    }
    results[slotno] = iter ? std::log2( maxiter-iter + 1 - std::log2(std::log2(dist) / 2)) * (4/std::log2(std::exp(1.))) : 0;
}

constexpr unsigned npixels = Xres * Yres, nthreads = 128, nblocks = (npixels + nthreads - 1) / nthreads;
constexpr unsigned num_streams = 2, num_threads = 2;

struct Task
{
    std::array<unsigned,npixels> pixels;

    virtual ~Task() { }
    virtual void Start(bool NeedMoment, double zr,double zi,double xscale,double yscale) = 0;
    virtual bool Running() const = 0;
    virtual bool Ready() const = 0;
    virtual bool End() = 0; // Returns new NeedMoment value
};

struct CudaTask: public Task
{
    std::uint16_t results[npixels], *p = nullptr;
    cudaStream_t stream;
    bool started = false;

    CudaTask()
    {
        checkCudaErrors(cudaMalloc((void**)&p, sizeof(results)));
        checkCudaErrors(cudaStreamCreate(&stream));
    }
    ~CudaTask()
    {
        if(Running()) End();
    }

    virtual void Start(bool NeedMoment, double zr,double zi,double xscale,double yscale)
    {
        if(NeedMoment)
            Iterate<true><<<nblocks, nthreads, 0, stream>>>( zr, zi, xscale, yscale, p);
        else
            Iterate<false><<<nblocks, nthreads, 0, stream>>>( zr, zi, xscale, yscale, p);

        cudaMemcpyAsync(results, p, sizeof(results), cudaMemcpyDeviceToHost, stream);
        started = true;
    }

    virtual bool Running() const
    {
        return started;
    }

    virtual bool Ready() const
    {
        return cudaStreamQuery(stream) != cudaErrorNotReady;
    }

    virtual bool End() // Returns new NeedMoment value
    {
        checkCudaErrors(cudaStreamSynchronize(stream));

        started = false;

        unsigned n_inside = std::count_if(results, results+npixels, std::bind1st(std::equal_to<std::uint16_t>(), 0));

        for(unsigned y=0; y<Yres; ++y)
            for(unsigned x=0; x<Xres; ++x)
                pixels[y*Xres + x] = Color(x,y, results[y*Xres+x]/8.);

        return n_inside >= (Xres*Yres)/1024;
    }
};

template<bool parallel>
extern bool SimdCalculation(bool NeedMoment, double zr,double zi,double xscale,double yscale, unsigned* pixels, unsigned index);
template<bool WithMoment>
extern double ThreadLoopHelperIterate(double zr, double zi);

bool ThreadCalculation(bool NeedMoment, double zr,double zi,double xscale,double yscale, unsigned* pixels)
{
    std::atomic<unsigned>    y_done{0}, n_inside{0};
    std::vector<std::thread> threads;
    for(unsigned n=0; n<8; ++n)
        threads.emplace_back([&](){
            unsigned count_inside = 0;
            for(unsigned y; (y = y_done++) < Yres; )
            {
                double i = zi+yscale*int(y-Yres/2);
                if(NeedMoment)
                    for(unsigned x=0; x<Xres; ++x)
                    {
                        double v = ThreadLoopHelperIterate<true>( zr+xscale*int(x-Xres/2), i );
                        if(v == 0.) ++count_inside;
                        pixels[y*Xres + x] = Color(x,y,v);
                    }
                else
                    for(unsigned x=0; x<Xres; ++x)
                    {
                        double v = ThreadLoopHelperIterate<false>( zr+xscale*int(x-Xres/2), i );
                        if(v == 0.) ++count_inside;
                        pixels[y*Xres + x] = Color(x,y,v);
                    }
            }
            n_inside += count_inside;
        });

    for(auto& t: threads) t.join();

    return n_inside >= (Xres*Yres)/1024;
}

struct NativeTask: public Task
{
    std::thread             thread;
    mutable std::mutex      lock;
    std::condition_variable task_available, task_finished;
    double zr=0,zi=0,xscale=0,yscale=0;
    bool   started=false, launched=false, finished=false, terminated=false, NeedMoment=false;

    NativeTask(unsigned index) : thread([this,index]()
    {
        std::unique_lock<std::mutex> lk(lock);
        for(;;)
        {
            task_available.wait(lk, [=]{ return terminated || started; });
            if(terminated) break;

            launched = true;
            started  = false;

            lk.unlock();

            if(true)//num_threads == 1)
            {
                NeedMoment = SimdCalculation<true>(NeedMoment, zr,zi,xscale,yscale, &pixels[0], index);
            }
            else
            {
                NeedMoment = SimdCalculation<false>(NeedMoment, zr,zi,xscale,yscale, &pixels[0], index);
                //NeedMoment = ThreadCalculation(NeedMoment, zr,zi,xscale,yscale, &pixels[0]);
            }


            lk.lock();
            finished = true;
            task_finished.notify_one();
        }
    })
    {
    }

    virtual ~NativeTask()
    {
        if(Running()) { End(); }
        { std::unique_lock<std::mutex> lk(lock);
        terminated = true;
        finished   = false; }
        task_available.notify_one();
        thread.join();
    }
    virtual void Start(bool mom, double r,double i,double xs,double ys)
    {
        { std::unique_lock<std::mutex> lk(lock);
        zr = r;
        zi = i;
        xscale = xs;
        yscale = ys;
        started = true;
        launched = false;
        finished = false;
        NeedMoment = mom; }
        task_available.notify_one();
    }
    virtual bool Running() const
    {
        return started || launched;
    }
    virtual bool Ready() const
    {
        return finished;
    }
    virtual bool End() // Returns new NeedMoment value
    {
        std::unique_lock<std::mutex> lk(lock);
        if(!finished) { task_finished.wait(lk,  [=]{return finished; }); }
        launched = false;
        return NeedMoment;
    }
};

int main()
{
    bool NeedMoment = true;

    constexpr unsigned num_tasks = num_streams + num_threads;
    std::array<std::unique_ptr<Task>, num_tasks> tasks;
    for(unsigned n=0; n<num_streams; ++n) tasks[n+          0] = std::unique_ptr<Task>(new CudaTask);
    for(unsigned n=0; n<num_threads; ++n) tasks[n+num_streams] = std::unique_ptr<Task>(new NativeTask(n));

    std::array<std::pair<unsigned,double>, num_tasks> info;
    std::map<unsigned,bool> flags;

    MAINLOOP_START(num_tasks * 4);

    while(MAINLOOP_GET_CONDITION_INFO())
    {
        // Wait until there is at least one free task
        bool started = false;
        for(unsigned t=0; t<num_tasks; ++t)
            if(!tasks[t]->Running())
            {
            runtask:;
                if(MAINLOOP_GET_CONDITION())
                {
                    double zr, zi, xscale, yscale; MAINLOOP_SET_COORDINATES_INFO(info[t]);
                    auto i = flags.lower_bound(info[t].first);
                    if(i != flags.begin()) NeedMoment = (--i)->second;
                    tasks[t]->Start(NeedMoment, zr, zi, xscale, yscale);
                    started = true;
                }
            }
            else if(tasks[t]->Ready())
            {
                NeedMoment = tasks[t]->End();
                flags[info[t].first] = NeedMoment;
                MAINLOOP_PUT_RESULT_INFO(tasks[t]->pixels, info[t], t);
                goto runtask;
            }

        // Wait a minimal time
        if(!started)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    MAINLOOP_FINISH();
}
