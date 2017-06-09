//static const unsigned Xres = 3840, Yres = 2160;
//static const unsigned Xres = 1920, Yres = 1080;
//static const unsigned Xres = 1706, Yres = 960;
static const unsigned Xres = 848, Yres = 480;
//static const unsigned Xres = 424, Yres = 240;
//static const unsigned Xres = 212, Yres = 120;

//#define TIMINGS_VERSION


#ifdef TIMINGS_VERSION
    static unsigned Color(unsigned x,unsigned y, double iter) { x=x; y=y; return int(iter); }
#else
    static unsigned Color(unsigned x,unsigned y, double iter)
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

#endif
