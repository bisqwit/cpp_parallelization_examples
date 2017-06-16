#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <cmath>
#include <algorithm>

#include <gd.h>
#include <cairo.h>

static constexpr char delimiter = ',';
static std::vector<std::string> split(const std::string& s)
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
static std::map<std::string, std::vector<double>> LoadTimings(const char* filename)
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

#include "blur.hh"

static int clamp(int value, int mi, int ma) { return std::max(std::min(value,ma), mi); }
static int ClampWithDesaturation(int r,int g,int b)
{
    int luma = r*2126 + g*7152 + b*722;
    if(luma > 2550000) { r=g=b=255; }
    else if(luma <= 0) { r=g=b=0; }
    else
    {
        double sat = 10000;
        if(r > 255) sat = std::min(sat, (luma-255e4) / (luma-r)); else if(r < 0) sat = std::min(sat, luma / (double)(luma-r));
        if(g > 255) sat = std::min(sat, (luma-255e4) / (luma-g)); else if(g < 0) sat = std::min(sat, luma / (double)(luma-g));
        if(b > 255) sat = std::min(sat, (luma-255e4) / (luma-b)); else if(b < 0) sat = std::min(sat, luma / (double)(luma-b));
        if(sat != 1.)
        {
            r = (r - luma) * sat/1e4 + luma; r = clamp(r,0,255);
            g = (g - luma) * sat/1e4 + luma; g = clamp(g,0,255);
            b = (b - luma) * sat/1e4 + luma; b = clamp(b,0,255);
        }
    }
    return unsigned(r)*65536u + unsigned(g)*256u + b;
}
static const unsigned prev_mul = 16;

template<unsigned W,unsigned H>
static void BloomPostprocess(unsigned short* prevpic, const unsigned* curpic, unsigned* targetpic)
{
    static short r[W*H], g[W*H], b[W*H];
    static short r2[W*H], g2[W*H], b2[W*H], r3[W*H], g3[W*H], b3[W*H];
    static short temp1[W*H], temp2[W*H], temp3[W*H], temp4[W*H], temp5[W*H], temp6[W*H];

    #pragma omp parallel for schedule(static)
    for(unsigned y=0; y<H; ++y)
        for(unsigned x=0; x<W; ++x)
        {
            unsigned r0 = prevpic[(y*W+x) * 3 + 0];
            unsigned g0 = prevpic[(y*W+x) * 3 + 1];
            unsigned b0 = prevpic[(y*W+x) * 3 + 2];
            unsigned current_pixel = curpic[y*W+x];
            unsigned r1 = (current_pixel >> 16) & 0xFF;
            unsigned g1 = (current_pixel >>  8) & 0xFF;
            unsigned b1 = (current_pixel >>  0) & 0xFF;
            unsigned rd = std::abs(int(r1 - r0/prev_mul));
            unsigned gd = std::abs(int(g1 - g0/prev_mul));
            unsigned bd = std::abs(int(b1 - b0/prev_mul));
            unsigned wanted_br = rd*rd + gd*gd + bd*bd;

            double diffpow = std::min(wanted_br, 32*32u) / double(32*32);
            double lumafactor = (r1*2126u + g1*7152u + b1*722u) / 2550000.;

            r[y*W+x] = r1 * std::pow(diffpow, 4.)*6 / lumafactor;
            g[y*W+x] = g1 * std::pow(diffpow, 4.)*6 / lumafactor;
            b[y*W+x] = b1 * std::pow(diffpow, 4.)*6 / lumafactor;

            constexpr unsigned hysteresis = 8;

            prevpic[(y*W+x) * 3 + 0] = (r0 * (hysteresis-1) + r1*prev_mul * 1) / hysteresis;
            prevpic[(y*W+x) * 3 + 1] = (g0 * (hysteresis-1) + g1*prev_mul * 1) / hysteresis;
            prevpic[(y*W+x) * 3 + 2] = (b0 * (hysteresis-1) + b1*prev_mul * 1) / hysteresis;
        }

    #pragma omp parallel sections
    {
        blur<3>(r, r2, temp1, W,H, 30.f);//std::hypot(W,H)/30.f);
        #pragma omp section
        blur<3>(g, g2, temp2, W,H, 30.f);//std::hypot(W,H)/30.f);
        #pragma omp section
        blur<3>(b, b2, temp3, W,H, 30.f);//std::hypot(W,H)/30.f);
        #pragma omp section
        blur<3>(r, r3, temp4, W,H, 140.f);//std::hypot(W,H)/30.f);
        #pragma omp section
        blur<3>(g, g3, temp5, W,H, 140.f);//std::hypot(W,H)/30.f);
        #pragma omp section
        blur<3>(b, b3, temp6, W,H, 140.f);//std::hypot(W,H)/30.f);
    }

    #pragma omp parallel for schedule(static)
    for(unsigned y=0; y<H; ++y)
        for(unsigned x=0; x<W; ++x)
        {
            unsigned p = curpic[y*W+x];
            unsigned r = ((p >> 16) & 0xFF) + r2[y*W+x] + r3[y*W+x];
            unsigned g = ((p >>  8) & 0xFF) + g2[y*W+x] + g3[y*W+x];
            unsigned b = ((p >>  0) & 0xFF) + b2[y*W+x] + b3[y*W+x];
            targetpic[y*W+x] = ClampWithDesaturation(r,g,b);
        }
}


int main()
{
    constexpr unsigned xres = 3840, yres = 2160;

    auto timings = LoadTimings("../timings_all.txt");
    timings.erase("Frame");

    std::initializer_list<std::pair<unsigned,std::string>> order{
        {0xFFFFFF, "vanilla" },
        {0xD070D0, "implicit-simd" },
        {0x40A0F2, "openmp-simd" },
        {0x80B4F6, "cilkplus-simd" },
        {0x70FFFF, "explicit-simd" }, //AACCFF
        {0x609000, "openmp-loop" },
        {0x20AA00, "cilkplus-loop" },
        {0xE0FFA0, "thread-loop" },
        {0xF00000, "openmp-offload" },
        {0x703000, "openacc-offload" },
        {0xAA3320, "cuda-offload" },
        {0xEE5540, "cuda-offload2" },
        {0xFFAA55, "cuda-offload3" }
    };

    static unsigned pixels[xres*yres];
    static unsigned short prev_pixels[xres*yres*3];
    static unsigned backup_pixels[xres*yres]{};
    std::fill_n(pixels, xres*yres, 0xFF000030u);

    auto SaveFrame = [](const std::string& filename, const unsigned* pixels)
    {
        fprintf(stderr, "Saving %s...\n", filename.c_str());
        gdImagePtr im = gdImageCreateTrueColor(xres, yres);
        ////BgdImageSaveAlpha(im, 1);
        gdImageAlphaBlending(im, 0);
        for(unsigned p=0, y=0; y<yres; ++y)
            for(unsigned x=0; x<xres; ++x, ++p)
                gdImageSetPixel(im, x,y, pixels[p]);// ^ 0xFF000000);

        std::FILE* fp = std::fopen(filename.c_str(), "wb");
        if(!fp) std::perror(filename.c_str());
        if(fp)
        {
            gdImagePng(im, fp);
            std::fclose(fp);
        }
        gdImageDestroy(im);
    };

    cairo_surface_t* sfc = cairo_image_surface_create_for_data
        ((unsigned char*)&pixels[0], CAIRO_FORMAT_ARGB32, xres, yres, xres*sizeof(unsigned));
    cairo_t* c   = cairo_create(sfc);
    cairo_set_antialias(c, CAIRO_ANTIALIAS_GRAY);

    auto AddAnimationFrame = [&](const std::string& which, unsigned n)
    {
    #if 1
        std::fprintf(stderr, "Processing... "); std::fflush(stderr);
        static unsigned newpixels[xres*yres];
        cairo_surface_flush(sfc);
        BloomPostprocess<xres,yres>(prev_pixels, pixels, newpixels);
    #else
        const auto* newpixels = pixels;
    #endif
        static unsigned framecounter=0;
        char Buf2[32]; std::sprintf(Buf2, "%05d-", framecounter++);
        char Buf[32]; std::sprintf(Buf, "-%04d.png", n);
        SaveFrame(Buf2+which+Buf, newpixels);
    };

    constexpr double   min_val = 50., max_val = 40000;
    constexpr unsigned line_region_height = 1920, line_region_start = 120;
    constexpr unsigned grid_region_width  = 3500, grid_region_start = 240;
    constexpr unsigned maxframe = 3200;

    auto ycoord = [=](double value)
    {
        value = value > 0 ? std::log(value) : 0;
        return (value - std::log(max_val)) * line_region_height / (std::log(min_val)-std::log(max_val)) + line_region_start;
    };
    auto xcoord = [=](unsigned value)
    {
        return grid_region_start + value * grid_region_width / maxframe;
    };
    auto Line = [&](unsigned x1,unsigned y1, unsigned x2,unsigned y2, int width=4)
    {
        cairo_new_path(c);
        cairo_stroke(c);
        cairo_set_line_width(c, width);
        cairo_move_to(c, x1,y1);
        cairo_line_to(c, x2,y2);
        cairo_stroke(c);
        cairo_close_path(c);
    };
    auto Hline = [&](unsigned x1,unsigned x2,unsigned y, int width=4) { Line(x1,y,x2,y, width); };
    auto Vline = [&](unsigned y1,unsigned y2,unsigned x, int width=4) { Line(x,y1,x,y2, width); };

    for(unsigned frame=0; frame<=maxframe; frame += 100)
    {
        cairo_set_source_rgb(c, .4, .3, .5); // Vline color
        Vline(line_region_start, line_region_start+line_region_height, xcoord(frame));

        if(frame%200==0)
        {
            cairo_select_font_face(c, "sans-serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
            cairo_set_font_size(c, 50.0);
            cairo_move_to(c, xcoord(frame) - 40 * int(1+std::log10(frame+0.))/2, line_region_start+line_region_height + 50);
            cairo_set_source_rgb(c, .7, .7, 1); // text color
            cairo_show_text(c, std::to_string(frame).c_str());
        }
    }

    cairo_select_font_face(c, "sans-serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(c, 50.0);
    cairo_set_source_rgb(c, .7, .7, 1); // text color
    cairo_move_to(c, 50, line_region_start/2);
    cairo_show_text(c, "Render time: Milliseconds per frame (less is better)");

    cairo_move_to(c, xres/2-40*6, yres-4);
    cairo_show_text(c, "Frame number");

    for(int val: {50,60,80,100,125,150,200,250,300,400,
                  500,600,800,1000,1250,1500,2000,2500,3000,4000,
                  5000,6000,8000,10000,12500,15000,20000,25000,30000,40000})
    {
        cairo_set_source_rgb(c, .4, .3, .5); // HLine color
        Hline(grid_region_start, grid_region_start+grid_region_width, ycoord(val));

        cairo_select_font_face(c, "sans-serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(c, 50.0);
        cairo_move_to(c, grid_region_start - 40*int(1+std::log10(val)), ycoord(val) + 25);
        cairo_set_source_rgb(c, .7, .7, 1); // text color
        cairo_show_text(c, std::to_string(val).c_str());
    }
    for(unsigned n=0; n<xres*yres; ++n)
    {
        unsigned pix = pixels[n];
        unsigned r = (pix>>16)&0xFF, g = (pix>>8)&0xFF, b = (pix>>0)&0xFF;
        prev_pixels[n*3+0] = r*prev_mul;
        prev_pixels[n*3+1] = g*prev_mul;
        prev_pixels[n*3+2] = b*prev_mul;
    }

    unsigned legend_ycoordinate = yres - 120 - 40*11;
    for(const auto& s: order)
    {
        //if(s.second == "openmp-offload" || s.second == "openacc-offload") continue;
        double width = 3.0;
        if(s.second == "cuda-offload")  width = 2.2;
        if(s.second == "cuda-offload2") width = 2.2;
        if(s.second == "cuda-offload3") width = 2.4;
        if(s.second == "cilkplus-loop") width = 2.4;
        if(s.second == "vanilla" || s.second == "explicit-simd") width = 5.0;
        if(s.second == "thread-loop") width = 1.5;
        if(s.second == "openmp-loop") width = 8;

        const double r = ((s.first >> 16) & 0xFF) / 255.;
        const double g = ((s.first >>  8) & 0xFF) / 255.;
        const double b = ((s.first >>  0) & 0xFF) / 255.;

        cairo_select_font_face(c, "sans-serif", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
        cairo_set_font_size(c, 35.0);
        cairo_move_to(c, xres-400, legend_ycoordinate);
        cairo_set_source_rgb(c, r,g,b);
        cairo_show_text(c, s.second.c_str());
        cairo_new_path(c);
        cairo_set_line_width(c, width);
        cairo_move_to(c, xres-440, legend_ycoordinate-10);
        cairo_line_to(c, xres-400, legend_ycoordinate-10);
        cairo_stroke(c);
        legend_ycoordinate += 40;

        cairo_surface_flush(sfc);
        std::copy_n(pixels+0, xres*yres, backup_pixels+0);

        const auto& tim = timings[s.second];
        bool prev = false;
        cairo_new_path(c);
        for(unsigned frame=0; frame <= (maxframe + 1000); ++frame)
        {
            double value = frame < tim.size() && frame <= maxframe ? tim[frame] : 0.;
            if(value == 0.)
            {
                prev = false;
            }
            else if(!prev)
            {
                cairo_move_to(c, xcoord(frame), ycoord(value));
                prev = true;
            }
            else
            {
                cairo_line_to(c, xcoord(frame), ycoord(value));
            }

            if((frame+1) % 5 == 0) //
            //if(frame == (maxframe+1000))
            {
                cairo_set_line_width(c, width);
                cairo_set_source_rgb(c, r,g,b);
                cairo_stroke_preserve(c);
                AddAnimationFrame(s.second, frame/5);
                std::copy_n(backup_pixels+0, xres*yres, pixels+0);
            }
        }
        cairo_set_line_width(c, width);
        cairo_set_source_rgb(c, r,g,b);
        cairo_stroke(c);
    }
    return 0;
}
