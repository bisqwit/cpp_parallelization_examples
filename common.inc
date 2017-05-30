#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <vector>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>
#include <array>
#include <cassert>

#include <SDL.h>
static const unsigned Xres = 848, Yres = 480;
//static const unsigned Xres = 424, Yres = 240;
//static const unsigned Xres = 212, Yres = 120;


double GetTime()
{
    static std::chrono::time_point<std::chrono::system_clock> begin = std::chrono::system_clock::now();
    return std::chrono::duration<double>( std::chrono::system_clock::now() - begin ).count();
}

struct Display
{
    SDL_Surface* s {SDL_SetVideoMode(Xres, Yres, 32,0)};
    double timeat[4]{GetTime(),GetTime(),GetTime(),GetTime()};
    unsigned frameat[4]{0,0,0,0}, frame{0}, interval=4, end=0;
    std::thread flipper;
public:
    Display() { flipper = std::thread([this]{while(!end){SDL_Flip(s);std::this_thread::sleep_for(std::chrono::milliseconds(1)); }}); }
    void Put(std::vector<unsigned>& pixels)
    {
        std::signal(SIGINT, SIG_DFL);
        std::memcpy(s->pixels, &pixels[0], 4*Xres*Yres);
        ++frame;
        double t = GetTime();
        int tog = (frame/interval)&3, mom=(frame%interval==0), tog2 = (frame/interval+2)&3;
        if(!mom) { frameat[tog]=frame; timeat[tog] = t; }
        std::printf("Frame%6u, %.2f fps...\r", frame, (frame-frameat[tog2]) / (t-timeat[tog2]));
        std::fflush(stdout);
    }
    ~Display()
    {
        end=1;
        flipper.join();
    }
} display;

unsigned Color(unsigned x,unsigned y, double iter)
{
    static unsigned char r[]{0x00,0x40,0x7E,0x11,0x16,0x38,0xFC,0xD0,0x5F,0xDC,0xFF,0x6B};
    static unsigned char g[]{0x00,0x40,0x9F,0x90,0x68,0xCF,0xFF,0x99,0x00,0x37,0x8E,0x14};
    static unsigned char b[]{0x00,0xE0,0xFF,0x9F,0x18,0x3F,0x00,0x24,0x09,0x0A,0xFE,0xBC};
    double d = ((((x&4)/4u + (x&2)*2u + (x&1)*16u) + (((x^y)&4)/2u + ((x^y)&2)*4u + ((x^y)&1)*32u))&0x3F)/64.;
    constexpr int k = 1;
    auto lerp = [d,k](int a,int b,double p) -> unsigned { return int(a/k + (b/k-a/k) * p + d)*255/(255/k); };
    return lerp(r[int(iter)%sizeof r], r[int(iter+1)%sizeof r], iter-int(iter))*0x10000u
         + lerp(g[int(iter)%sizeof r], g[int(iter+1)%sizeof r], iter-int(iter))*0x100u
         + lerp(b[int(iter)%sizeof r], b[int(iter+1)%sizeof r], iter-int(iter))*0x1u;
}
