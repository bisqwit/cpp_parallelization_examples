#include "common.inc"

#include <x86intrin.h>

#include "simd_emu.hh"

__m256i to256(__mmask8 c)
    { return _mm256_mask_mov_epi32(_mm256_set1_epi32(-1), c, _mm256_setzero_si256()); }
__m512d ifelse(__m256i c, __m512d ok, __m512d nok)
    { return _mm512_mask_mov_pd(nok, _mm256_cmp_epi32_mask(c, _mm256_setzero_si256(), _MM_CMPINT_NE), ok); }


__m128i to128(__m256d v)
    { __m256i c = _mm256_castpd_si256(v); return _mm_packs_epi16(extract128(c,0), extract128(c,1)); }
__m256d ifelse(__m256i c, __m256d ok, __m256d nok) { return _mm256_blendv_pd(nok, ok, _mm256_castsi256_pd(c)); }
__m256d ifelse(__m128i c, __m256d ok, __m256d nok) { return ifelse(_mm256_cvtepi32_epi64(c), ok, nok); }


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

__m256d Iterate(__m256d zr, __m256d zi)
{
    const double escape_radius_squared = ESCAPE_RADIUS_SQUARED;
    const int maxiter = MAXITER;
    __m256d cr = zr, sr = cr;
    __m256d ci = zi, si = ci;
    __m256d dist = _mm256_set1_pd(0.0), limit = _mm256_set1_pd(escape_radius_squared);

    __m128i notescaped = _mm_set1_epi32(-1), one = _mm_set1_epi32(1), zero = _mm_setzero_si128();
    __m128i iter       = _mm_set1_epi32(maxiter);

    while(!_mm_testz_si128(notescaped, notescaped))
    {
        auto i2    = _mm256_mul_pd(ci,ci);
        dist       = ifelse(notescaped, _mm256_fmadd_pd(cr,cr, i2), dist);

        notescaped = _mm_and_si128(notescaped,
                     _mm_andnot_si128(_mm_cmpeq_epi32(iter, zero),
                                      to128(_mm256_cmp_pd(dist, limit, _CMP_LT_OQ))));
        iter       = _mm_add_epi32(iter, notescaped);

        ci = _mm256_fmadd_pd(_mm256_mul_pd(cr, ci), _mm256_set1_pd(2), zi);
        cr = _mm256_fmadd_pd(cr,cr, _mm256_sub_pd(zr, i2));

        __m128i moment = _mm_cmpeq_epi32(_mm_and_si128(iter, _mm_sub_epi32(iter, one)), zero);

        iter = _mm_andnot_si128(to128(_mm256_and_pd(_mm256_cmp_pd(cr,sr,_CMP_EQ_OQ),
                                                    _mm256_cmp_pd(ci,si,_CMP_EQ_OQ))), iter);
        sr = ifelse(moment, cr, sr);
        si = ifelse(moment, ci, si);
    }
    return _mm256_castsi256_pd(_mm256_andnot_si256(_mm256_cvtepi32_epi64(_mm_cmpeq_epi32(iter, zero)), _mm256_castpd_si256(
           _mm256_mul_pd(_mm256_set1_pd(4/std::log2(std::exp(1.))),
                         _mm256_log2_pd(_mm256_add_pd(_mm256_cvtepi32_pd(_mm_sub_epi32(_mm_set1_epi32(maxiter), iter)),
                                                      _mm256_sub_pd(_mm256_set1_pd(1),
                                                                    _mm256_log2_pd(_mm256_mul_pd(_mm256_log2_pd(dist),
                                                                                                 _mm256_set1_pd(0.5))))))))));
}

int main()
{
    while(GetTime() < 5)
    {
        std::vector<unsigned> pixels (Xres * Yres);

        double zr = -0.743639266077433, zi = 0.131824786875559, scale = 4. * std::pow(2, -std::min(GetTime(),53.)*0.7);
        double xscale = scale/Yres, yscale = -scale/Yres;
        #if defined(__AVX2__) || defined(__AVX512F__)
        constexpr unsigned N=8;
        #else
        constexpr unsigned N=4;
        #endif

        switch(N)
        {
            case 4:
                for(unsigned y=0; y<Yres; ++y)
                {
                    __m256d i = _mm256_set1_pd( zi+yscale*int(y-Yres/2) );

                    for(unsigned x=0; x<Xres/N*N; x += N)
                    {
                        __m256d r = _mm256_fmadd_pd(_mm256_set1_pd(xscale), _mm256_add_pd(_mm256_set_pd(3,2,1,0),
                                                                                          _mm256_set1_pd(int(x-Xres/2))),
                                                    _mm256_set1_pd(zr));

                        __m256d results = Iterate(r,i);
                        for(unsigned n=0; n<N; ++n) { pixels[y*Xres + x+n] = Color(x+n,y, results[n]); }
                    }
                }
                break;

            case 8:
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
                break;
        }

        display.Put(pixels);
    }
    std::printf("\n%u frames rendered\n", display.frame);
}
