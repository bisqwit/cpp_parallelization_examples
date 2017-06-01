#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
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
static const unsigned Xres = 3840, Yres = 2160;
//static const unsigned Xres = 1920, Yres = 1080;
//static const unsigned Xres = 1706, Yres = 960;
//static const unsigned Xres = 848, Yres = 480;
//static const unsigned Xres = 424, Yres = 240;
//static const unsigned Xres = 212, Yres = 120;

double GetTime()
{
    static std::chrono::time_point<std::chrono::system_clock> begin = std::chrono::system_clock::now();
    return std::chrono::duration<double>( std::chrono::system_clock::now() - begin ).count();
}

#if 1
#include <map>
#include <string>
#include <fstream>
#include <iomanip>
/*#include <regex>*/
#include <setjmp.h>

static constexpr char delimiter = ',';

std::vector<std::string> split(const std::string& s)
{
    // passing -1 as the submatch index parameter performs splitting
  /*
    std::regex t(",");
    return {std::sregex_token_iterator(s.begin(), s.end(), t, -1),
            std::sregex_token_iterator{}};
  */
    std::size_t begin = 0;
    std::vector<std::string> result;
    while(begin != s.size())
    {
        std::size_t p = s.find(delimiter, begin);
        if(p == s.npos) { result.emplace_back(s.begin() + begin, s.end()); break; }
        result.emplace_back(s.begin() + begin, s.begin() + p);
        begin = p+1;
    }
    return result;
}
std::map<std::string, std::vector<double>> LoadTimings(const char* filename)
{
    std::vector<std::string> headers;
    std::map<std::string, std::vector<double>> result;
    std::ifstream f(filename);
    std::string line;
    while(std::getline(f, line))
    {
        if(headers.empty())
        {
            headers = split(line);
            for(const auto& h: headers) { result[h]; }
        }
        else
        {
            unsigned h=0;
            for(auto& s: split(line))
            {
                if(h > headers.size()) break;
                result[headers[h]].push_back(std::stod(s));
                ++h;
            }
        }
    }
    return result;
}

unsigned Color(unsigned x,unsigned y, double iter) { x=x; y=y; return int(iter); }

jmp_buf termination;

volatile bool Terminated = false;
void Terminate(int sig)
{
    std::signal(sig, SIG_DFL);
    std::fprintf(stderr, "\nSIG %d\n", sig);
    Terminated = true;
    longjmp(termination, 1);
}

    #define MAINLOOP_START() \
            static unsigned framecount = 0; \
            double   start_time = GetTime(), prev_frame_end_time = start_time; \
            auto timings  = LoadTimings("timings_all.txt"); \
            timings.erase("Frame"); \
            std::ofstream f("timings_all.txt"); \
            auto& tab = timings[PROG_NAME]; \
            f << "Frame" << delimiter; for(const auto& h: timings) f << h.first << delimiter; \
            f << std::endl << std::flush; \
            if(setjmp(termination) > 0) goto done_loop; \
            std::signal(SIGINT, Terminate); \
            std::signal(SIGTERM, Terminate); \
            std::signal(SIGSEGV, Terminate);

    #define MAINLOOP_GET_CONDITION()    \
            framecount < 3600 && !Terminated

    #define MAINLOOP_SET_COORDINATES()  \
            zr = -0.743639266077433; \
            zi = +0.131824786875559; \
            double scale = 4. * std::pow(2, -std::min(framecount / 60., 53.)*0.7); \
            xscale = scale / Yres; \
            yscale = scale / Yres;

    #define MAINLOOP_PUT_RESULT(pixels) do { \
                double frame_end      = GetTime(); \
                double frame_duration = 1e3 * ((frame_end - start_time) - (prev_frame_end_time - start_time)); \
                std::printf("\rFrame %u: %.8f ms\n", framecount, frame_duration); std::fflush(stdout); \
                if(tab.size() <= framecount) \
                    tab.push_back(frame_duration); \
                else if(tab[framecount] == 0.) \
                    tab[framecount] = frame_duration; \
                else \
                    tab[framecount] = std::min(frame_duration, tab[framecount]); \
                f << (framecount+1) << delimiter; \
                for(const auto& h: timings) f << std::setprecision(20) << (h.second.size() <= framecount ? 0. : h.second[framecount]) << delimiter; \
                f << std::endl << std::flush; \
                ++framecount; \
                prev_frame_end_time = frame_end; \
            } while(0)

    #define MAINLOOP_FINISH() do { \
            done_loop:; \
                std::printf("\n%u frames rendered in %g seconds\n", framecount, GetTime()-start_time); \
                unsigned maxframe = 0; \
                for(const auto& h: timings) maxframe = std::max(maxframe, unsigned(h.second.size())); \
                while(framecount < maxframe) \
                { \
                    f << (framecount+1) << delimiter; \
                    for(const auto& h: timings) f << std::setprecision(20) << (h.second.size() <= framecount ? 0. : h.second[framecount]) << delimiter; \
                    f << std::endl << std::flush; \
                    ++framecount; \
                } \
            } while(0)

#else

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
    constexpr int k = 1, m = 0x3F;
    //constexpr int k = 96, m = 0x30;
    double d = ((((x&4)/4u + (x&2)*2u + (x&1)*16u) + (((x^y)&4)/2u + ((x^y)&2)*4u + ((x^y)&1)*32u))&m)/64.;
    auto lerp = [d,k](int a,int b,double p) -> unsigned { return int(a/k + (b/k-a/k) * p + d)*255/(255/k); };
    return lerp(r[int(iter)%sizeof r], r[int(iter+1)%sizeof r], iter-int(iter))*0x10000u
         + lerp(g[int(iter)%sizeof r], g[int(iter+1)%sizeof r], iter-int(iter))*0x100u
         + lerp(b[int(iter)%sizeof r], b[int(iter+1)%sizeof r], iter-int(iter))*0x1u;
}

    #define MAINLOOP_START()             /**/
    #define MAINLOOP_GET_CONDITION()     GetTime() < 5
    #define MAINLOOP_SET_COORDINATES()   do { \
            zr = -0.743639266077433; \
            zi = +0.131824786875559; \
            double scale = 4. * std::pow(2, -std::min(GetTime(),53.)*0.7); \
            xscale = scale / Yres; \
            yscale = scale / Yres; } while(0)

    #define MAINLOOP_PUT_RESULT(pixels) display.Put(pixels)
    #define MAINLOOP_FINISH()           std::printf("\n%u frames rendered\n", display.frame)
#endif
