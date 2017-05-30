

#include <SDL.h>
static const unsigned Xres = 320, Yres = 240;

double GetTime()
{
    static std::chrono::time_point<std::chrono::system_clock> begin = std::chrono::system_clock::now();
    return std::chrono::duration<double>( std::chrono::system_clock::now() - begin );
}

class Display
{
    SDL_Window*   w = SDL_CreateWindow("zoom", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, Xres, Yres, SDL_WINDOW_RESIZABLE);
    SDL_Renderer* r = SDL_CreateRenderer(w, -1, 0);
    SDL_Texture*  t = SDL_CreateTexture(r, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, Xres, Yres);
    unsigned frame = 0;
public:
    void Put(const std::vector<unsigned>& pixels)
    {
        SDL_UpdateTexture(t, nullptr, &pixels[0], 4*Xres);
        SDL_RenderCopy(r, t, nullptr, nullptr);
        SDL_RenderPresent(r);
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
