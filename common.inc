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
#include <condition_variable>
#include <mutex>
#include <utility>

#include <SDL.h>

#include "defs.inc"

double GetTime()
{
    static std::chrono::time_point<std::chrono::system_clock> begin = std::chrono::system_clock::now();
    return std::chrono::duration<double>( std::chrono::system_clock::now() - begin ).count();
}

#ifdef TIMINGS_VERSION
#include <map>
#include <set>
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

jmp_buf termination;

volatile bool Terminated = false;
void Terminate(int sig)
{
    std::signal(sig, SIG_DFL);
    std::fprintf(stderr, "\nSIG %d\n", sig);
    Terminated = true;
    longjmp(termination, 1);
}

    #define MAINLOOP_START(queue_length) \
            static unsigned frames_done = 0, frames_printed = 0, frames_timestamped = 0; \
            constexpr unsigned queue_size = (queue_length); \
            std::set<unsigned> frames_finished; \
            double   start_time = GetTime(), prev_frame_end_time[queue_size]; \
            std::map<unsigned, double> all_end_times; \
            for(auto& d: prev_frame_end_time) d = start_time; \
            auto timings  = LoadTimings("timings_all.txt"); \
            timings.erase("Frame"); \
            std::ofstream f("timings_all.txt"); \
            auto& tab = timings[PROG_NAME]; \
            f << "Frame" << delimiter; for(const auto& h: timings) f << h.first << delimiter; \
            if(PROG_NAME == std::string("cuda-offload3")) { tab.clear(); timings.erase("cuda-offload3.o"); } \
            f << std::endl << std::flush; \
            if(setjmp(termination) > 0) goto done_loop; \
            std::signal(SIGINT, Terminate); \
            std::signal(SIGTERM, Terminate); \
            std::signal(SIGSEGV, Terminate);

    #define MAINLOOP_GET_CONDITION()    \
            frames_done < 3600 && !Terminated

    #define MAINLOOP_GET_CONDITION_INFO() \
            (frames_done < 3600 || frames_finished.size() < frames_done) && !Terminated

    #define MAINLOOP_DO_SET_COORDINATES(frameno)  \
            zr = -0.743639266077433; \
            zi = +0.131824786875559; \
            double scale = 4. * std::pow(2, -std::min(frameno / 60., 53.)*0.7); \
            xscale = scale / Yres; \
            yscale = scale / Yres;

    #define MAINLOOP_DO_PUT_RESULT(pixels, frameno, duration) do { \
                unsigned whichframe = frameno; \
                if(tab.size() <= whichframe) tab.resize(whichframe+1); \
                if(tab[whichframe] == 0.) \
                    tab[whichframe] = duration; \
                else \
                    tab[whichframe] = std::min(duration, tab[whichframe]); \
                \
                frames_finished.insert(whichframe); \
                while(frames_finished.find(frames_printed) != frames_finished.end()) \
                { \
                    f << (frames_printed+1) << delimiter; \
                    for(const auto& h: timings) f << std::setprecision(20) << (h.second.size() <= frames_printed ? 0. : h.second[frames_printed]) << delimiter; \
                    f << std::endl << std::flush; \
                    ++frames_printed; \
                } \
            } while(0)

    #define MAINLOOP_SET_COORDINATES() \
            MAINLOOP_DO_SET_COORDINATES(frames_done)

    #define RECORD_DURATION(frameno, duration) do { \
                if(tab.size() <= frameno) tab.resize(frameno+1); \
                if(tab[frameno] == 0.) \
                    tab[frameno] = duration; \
                else \
                    tab[frameno] = std::min(duration, tab[frameno]); \
            } while(0)

    #define PRINT_FRAME(frameno) do { \
                f << (frameno+1) << delimiter; \
                for(const auto& h: timings) f << std::setprecision(20) << (h.second.size() <= frameno ? 0. : h.second[frameno]) << delimiter; \
                f << std::endl << std::flush; } while(0)

    #define MAINLOOP_PUT_RESULT(pixels) do { \
            double frame_end      = GetTime(); \
            double frame_duration = 1e3/queue_size * ((frame_end - start_time) - (prev_frame_end_time[frames_done % queue_size] - start_time)); \
            std::printf("\rFrame %u: %.8f ms\n", frames_done, frame_duration); std::fflush(stdout); \
            RECORD_DURATION(frames_done, frame_duration); \
            \
            frames_finished.insert(frames_done); \
            while(frames_finished.find(frames_printed) != frames_finished.end()) \
            { \
                PRINT_FRAME(frames_printed); \
                ++frames_printed; \
            } \
            prev_frame_end_time[frames_done++ % queue_size] = frame_end; } while(0) \

    #define MAINLOOP_SET_COORDINATES_INFO(info)  \
            info.first  = frames_done++; \
            info.second = GetTime(); \
            MAINLOOP_DO_SET_COORDINATES(info.first)

    #define MAINLOOP_PUT_RESULT_INFO(pixels, info, taskno) do { \
            double frame_end      = GetTime(); \
            all_end_times[info.first] = frame_end; \
            double frame_duration = 1e3/std::min(queue_size, frames_timestamped+1) * ((frame_end - start_time) - (prev_frame_end_time[frames_timestamped % queue_size] - start_time)); \
            RECORD_DURATION(info.first, frame_duration); \
            std::printf("\rFrame %u/%u: %.8f ms but recording %.8f ms #%u\n", \
                info.first, frames_done, 1e3 * (frame_end - info.second), frame_duration, taskno); \
            std::fflush(stdout); \
            frames_finished.insert(info.first); \
            while(frames_finished.find(frames_printed) != frames_finished.end()) { \
                /*double frame_end_at      = all_end_times.find(frames_printed)->second; \
                double prev_frame_end_at = frames_printed > 0 ? all_end_times.find(frames_printed-1)->second : start_time; \
                double frame_duration    = 1e3 * ((frame_end_at - start_time) - (prev_frame_end_at - start_time)); \
                RECORD_DURATION(frames_printed, frame_duration); \
                std::printf("\rFrame %u: actually %.8f ms\n", frames_printed, frame_duration);*/ \
                PRINT_FRAME(frames_printed); \
                ++frames_printed; \
            } \
            prev_frame_end_time[frames_timestamped++ % queue_size] = frame_end; } while(0) \

    #define MAINLOOP_FINISH() do { \
            done_loop:; \
                std::printf("\n%u frames rendered in %g seconds\n", frames_done, GetTime()-start_time); \
                unsigned maxframe = 0; \
                for(const auto& h: timings) maxframe = std::max(maxframe, unsigned(h.second.size())); \
                while(frames_printed < maxframe) \
                { \
                    PRINT_FRAME(frames_printed); \
                    ++frames_printed; \
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

    #define MAINLOOP_START(n)            /**/
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
