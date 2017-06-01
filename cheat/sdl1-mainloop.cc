

#include <SDL.h>
static const unsigned Xres = 320, Yres = 240;

double GetTime()
{
    static std::chrono::time_point<std::chrono::system_clock> begin = std::chrono::system_clock::now();
    return std::chrono::duration<double>( std::chrono::system_clock::now() - begin );
}

class Display
{
    SDL_Surface*  s = SDL_SetVideoMode(Xres, Yres, 32,0);
    unsigned frame = 0;
public:
    void Put(const std::vector<unsigned>& pixels)
    {
        std::memcpy(s->pixels, &pixels[0], 4*Xres*Yres);
        SDL_Flip(s);
        ++frame;
        std::fprintf(stderr, "Frame%6u, %.2f fps...\r", frame, GetTime() / frame);
        std::fflush(stderr);
    }
};

int main()
{
    Display d;
    for(;;)
    {
        std::vector<unsigned> pixels (Xres * Yres);
        d.Put(pixels);
    }
}
