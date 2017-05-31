#include "common.inc"

#include <x86intrin.h>

#include "simd_emu.hh"

__m256i to256(__mmask8 c)
    { return _mm256_mask_mov_epi32(_mm256_set1_epi32(-1), c, _mm256_setzero_si256()); }
__m512d ifelse(__m256i c, __m512d ok, __m512d nok)
    { return _mm512_mask_mov_pd(nok, _mm256_cmp_epi32_mask(c, _mm256_setzero_si256(), _MM_CMPINT_NE), ok); }

__m512d Iterate(__m512d zr, __m512d zi)
{
    const double escape_radius_squared = ESCAPE_RADIUS_SQUARED;
    const int maxiter = MAXITER;
    __m512d cr = zr, sr = cr;
    __m512d ci = zi, si = ci;
    __m512d dist = _mm512_set1_pd(0.0), limit = _mm512_set1_pd(escape_radius_squared);

    __m256i notescaped = _mm256_set1_epi32(-1), one  = _mm256_set1_epi32(1), zero = _mm256_setzero_si256();
    __m256i iter = _mm256_set1_epi32(maxiter);

    while(!_mm256_testz_si256(notescaped, notescaped))
    {
        auto i2    = _mm512_mul_pd(ci,ci);
        dist       = ifelse(notescaped, _mm512_fmadd_pd(cr,cr, i2), dist);

        notescaped = _mm256_and_si256(notescaped,
                     _mm256_andnot_si256(_mm256_cmpeq_epi32(iter, zero),
                                         to256(_mm512_cmp_pd_mask(dist, limit, _MM_CMPINT_LT))));
        iter       = _mm256_add_epi32(iter, notescaped);

        ci = _mm512_fmadd_pd(_mm512_mul_pd(cr, ci), _mm512_set1_pd(2), zi);
        cr = _mm512_fmadd_pd(cr,cr, _mm512_sub_pd(zr, i2));

        __m256i moment = _mm256_cmpeq_epi32(_mm256_and_si256(iter, _mm256_sub_epi32(iter, one)), zero);

        iter = _mm256_andnot_si256(to256(_mm512_kand(_mm512_cmp_pd_mask(cr,sr,_MM_CMPINT_EQ),
                                                     _mm512_cmp_pd_mask(ci,si,_MM_CMPINT_EQ))), iter);
        sr = ifelse(moment, cr, sr);
        si = ifelse(moment, ci, si);
    }
    return _mm512_castsi512_pd(_mm512_andnot_si512(_mm512_cvtepi32_epi64(_mm256_cmpeq_epi32(iter, zero)), _mm512_castpd_si512(
           _mm512_mul_pd(_mm512_set1_pd(4/std::log2(std::exp(1.))),
                         _mm512_log2_pd(_mm512_add_pd(_mm512_cvtepi32_pd(_mm256_sub_epi32(_mm256_set1_epi32(maxiter), iter)),
                                                      _mm512_sub_pd(_mm512_set1_pd(1),
                                                                    _mm512_log2_pd(_mm512_mul_pd(_mm512_log2_pd(dist),
                                                                                                 _mm512_set1_pd(0.5))))))))));
}

int main()
{
    while(GetTime() < 5)
    {
        std::vector<unsigned> pixels (Xres * Yres);

        double zr = -0.743639266077433, zi = 0.131824786875559, scale = 4. * std::pow(2, -std::min(GetTime(),53.)*0.7);
        double xscale = scale/Yres, yscale = -scale/Yres;
        constexpr unsigned N=8;

        for(unsigned y=0; y<Yres; ++y)
        {
            __m512d i = _mm512_set1_pd( zi+yscale*int(y-Yres/2) );

            for(unsigned x=0; x<Xres/N*N; x += N)
            {
                __m512d r = _mm512_fmadd_pd(_mm512_set1_pd(xscale), _mm512_add_pd(_mm512_set_pd(7,6,5,4,3,2,1,0),
                                                                                  _mm512_set1_pd(int(x-Xres/2))),
                                            _mm512_set1_pd(zr));

                __m512d results = Iterate(r,i);
                for(unsigned n=0; n<N; ++n) { pixels[y*Xres + x+n] = Color(x+n,y, results[n]); }
            }
        }

        display.Put(pixels);
    }
    std::printf("\n%u frames rendered\n", display.frame);
}
