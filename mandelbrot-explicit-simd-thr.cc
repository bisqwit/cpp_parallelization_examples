#include "common.inc"

#include <x86intrin.h>
#include "simd_emu.hh"

inline __m512d _mm512_log2_pd(__m512d x) /* log2(x) for eight positive doubles */
{
    constexpr int mantissa_bits = 52, exponent_bias = 1022;
    __m512d half = _mm512_set1_pd(0.5);
    __m512i e = _mm512_srli_epi64(_mm512_castpd_si512(x), mantissa_bits);
    __m512i m = _mm512_and_si512(_mm512_castpd_si512(x), _mm512_set1_epi64((1ull << mantissa_bits)-1));
    x = _mm512_or_pd(half, _mm512_castsi512_pd(m));

    __mmask8 lt = _mm512_cmp_pd_mask(x, _mm512_set1_pd(1/std::sqrt(2.)), _MM_CMPINT_LT);
    __m512d ltid = _mm512_castsi512_pd(_mm512_mask_mov_epi64(_mm512_set1_epi64(0), lt, _mm512_set1_epi64(-1)));
    __m512i lti = _mm512_castpd_si512(ltid);
    __m512d dbl_e = _mm512_sub_pd(_mm512_cvtepi64_pd(_mm512_add_epi64(e,lti)), _mm512_set1_pd(exponent_bias));

    __m512d z = _mm512_sub_pd(x, _mm512_add_pd(half, _mm512_andnot_pd(ltid, half)));
    __m512d y = _mm512_fmadd_pd(half, _mm512_sub_pd(x, _mm512_and_pd(ltid, half)), half);
    x = _mm512_div_pd(z, y);
    z = _mm512_mul_pd(x, x);
    __m512d u = _mm512_add_pd(z, _mm512_set1_pd(-3.56722798512324312549E1));
    __m512d t =                  _mm512_set1_pd(-7.89580278884799154124E-1);
    u = _mm512_fmadd_pd(u, z, _mm512_set1_pd(3.12093766372244180303E2));
    t = _mm512_fmadd_pd(t, z, _mm512_set1_pd(1.63866645699558079767E1));
    u = _mm512_fmadd_pd(u, z, _mm512_set1_pd(-7.69691943550460008604E2));
    t = _mm512_fmadd_pd(t, z, _mm512_set1_pd(-6.41409952958715622951E1));
    y = _mm512_fmadd_pd(z, _mm512_div_pd(t, u), _mm512_add_pd(half,half));
    return _mm512_fmadd_pd(x, _mm512_mul_pd(y, _mm512_set1_pd(std::log2(std::exp(1.)))), dbl_e);
}

__m256i to256(__mmask8 c)
    { return _mm256_mask_mov_epi32(_mm256_set1_epi32(-1), c, _mm256_setzero_si256()); }
__m512d ifelse(__m256i c, __m512d ok, __m512d nok)
    { return _mm512_mask_mov_pd(nok, _mm256_cmp_epi32_mask(c, _mm256_setzero_si256(), _MM_CMPINT_NE), ok); }

template<bool WithMoment>
__m512d Iterate(__m512d zr, __m512d zi)
{
    const double escape_radius_squared = ESCAPE_RADIUS_SQUARED;
    const int maxiter = MAXITER;
    __m512d cr = zr, sr = cr;
    __m512d ci = zi, si = ci;
    __m512d dist = _mm512_set1_pd(0.0), limit = _mm512_set1_pd(escape_radius_squared);

    __m256i one  = _mm256_set1_epi32(1), zero = _mm256_setzero_si256();
    __m512d i2   = _mm512_mul_pd(ci,ci), r2 = _mm512_mul_pd(cr,cr);

    __m256i notescaped = to256(_mm512_kand(
                               _mm512_cmp_pd_mask(
                                   _mm512_fmadd_pd(i2,
                                       _mm512_fmadd_pd(i2, _mm512_set1_pd(8),
                                                       _mm512_fmadd_pd(r2, _mm512_set1_pd(16), _mm512_set1_pd(-3))),
                                       _mm512_fmadd_pd(r2, _mm512_fmadd_pd(r2, _mm512_set1_pd(8), _mm512_set1_pd(-3)), cr)),
                                   _mm512_set1_pd(3./32), _MM_CMPINT_GE),
                               _mm512_cmp_pd_mask(
                                   _mm512_fmadd_pd(_mm512_add_pd(cr,_mm512_set1_pd(1)),_mm512_add_pd(cr,_mm512_set1_pd(1)), i2),
                                   _mm512_set1_pd(1./16), _MM_CMPINT_GE)));

    __m256i iter = _mm256_and_si256(_mm256_set1_epi32(maxiter), notescaped);

    while(!_mm256_testz_si256(notescaped, notescaped))
    {
        dist       = ifelse(notescaped, _mm512_fmadd_pd(cr,cr, i2), dist);

        notescaped = _mm256_and_si256(notescaped,
                     _mm256_andnot_si256(_mm256_cmpeq_epi32(iter, zero),
                                         to256(_mm512_cmp_pd_mask(dist, limit, _MM_CMPINT_LT))));
        iter       = _mm256_add_epi32(iter, notescaped);

        ci = _mm512_fmadd_pd(_mm512_mul_pd(cr, ci), _mm512_set1_pd(2), zi);
        cr = _mm512_fmadd_pd(cr,cr, _mm512_sub_pd(zr, i2));
        i2 = _mm512_mul_pd(ci,ci);

        if(WithMoment)
        {
            __m256i moment = _mm256_cmpeq_epi32(_mm256_and_si256(iter, _mm256_sub_epi32(iter, one)), zero);

            iter = _mm256_andnot_si256(to256(_mm512_kand(_mm512_cmp_pd_mask(cr,sr,_MM_CMPINT_EQ),
                                                         _mm512_cmp_pd_mask(ci,si,_MM_CMPINT_EQ))), iter);
            sr = ifelse(moment, cr, sr);
            si = ifelse(moment, ci, si);
        }
    }
    return _mm512_castsi512_pd(_mm512_andnot_si512(_mm512_cvtepi32_epi64(_mm256_cmpeq_epi32(iter, zero)), _mm512_castpd_si512(
           _mm512_mul_pd(_mm512_set1_pd(4/std::log2(std::exp(1.))),
                         _mm512_log2_pd(_mm512_add_pd(_mm512_cvtepi32_pd(_mm256_sub_epi32(_mm256_set1_epi32(maxiter), iter)),
                                                      _mm512_sub_pd(_mm512_set1_pd(1),
                                                                    _mm512_log2_pd(_mm512_mul_pd(_mm512_log2_pd(dist),
                                                                                                 _mm512_set1_pd(0.5))))))))));
}



inline __m256d _mm256_log2_pd(__m256d x) /* log2(x) for four positive doubles */
{
    constexpr int mantissa_bits = 52, exponent_bias = 1022;
    __m256d half = _mm256_set1_pd(0.5);
    __m256i e = _mm256_srli_epi64(_mm256_castpd_si256(x), mantissa_bits);
    __m256i m = _mm256_and_si256(_mm256_castpd_si256(x), _mm256_set1_epi64x((1ull << mantissa_bits)-1));
    x = _mm256_or_pd(half, _mm256_castsi256_pd(m));

    __m256d ltid = _mm256_cmp_pd(x, _mm256_set1_pd(1/std::sqrt(2.)), _CMP_LT_OQ);
    __m256i lti = _mm256_castpd_si256(ltid);
    __m256d dbl_e = _mm256_sub_pd(_mm256_cvtepi64_pd(_mm256_add_epi64(e,lti)), _mm256_set1_pd(exponent_bias));

    __m256d z = _mm256_sub_pd(x, _mm256_add_pd(half, _mm256_andnot_pd(ltid, half)));
    __m256d y = _mm256_fmadd_pd(half, _mm256_sub_pd(x, _mm256_and_pd(ltid, half)), half);
    x = _mm256_div_pd(z, y);
    z = _mm256_mul_pd(x, x);
    __m256d u = _mm256_add_pd(z, _mm256_set1_pd(-3.56722798256324312549E1));
    __m256d t =                  _mm256_set1_pd(-7.89580278884799154124E-1);
    u = _mm256_fmadd_pd(u, z, _mm256_set1_pd(3.12093766372244180303E2));
    t = _mm256_fmadd_pd(t, z, _mm256_set1_pd(1.63866645699558079767E1));
    u = _mm256_fmadd_pd(u, z, _mm256_set1_pd(-7.69691943550460008604E2));
    t = _mm256_fmadd_pd(t, z, _mm256_set1_pd(-6.41409952958715622951E1));
    y = _mm256_fmadd_pd(z, _mm256_div_pd(t, u), _mm256_add_pd(half,half));
    return _mm256_fmadd_pd(x, _mm256_mul_pd(y, _mm256_set1_pd(std::log2(std::exp(1.)))), dbl_e);
}

__m128i to128(__m256d v)
    { __m256i c = _mm256_castpd_si256(v); return _mm_packs_epi16(extract128(c,0), extract128(c,1)); }
__m256d ifelse(__m256i c, __m256d ok, __m256d nok) { return _mm256_blendv_pd(nok, ok, _mm256_castsi256_pd(c)); }
__m256d ifelse(__m128i c, __m256d ok, __m256d nok) { return ifelse(_mm256_cvtepi32_epi64(c), ok, nok); }

template<bool WithMoment>
__m256d Iterate(__m256d zr, __m256d zi)
{
    const double escape_radius_squared = ESCAPE_RADIUS_SQUARED;
    const int maxiter = MAXITER;
    __m256d cr = zr, sr = cr;
    __m256d ci = zi, si = ci;
    __m256d dist = _mm256_set1_pd(0.0), limit = _mm256_set1_pd(escape_radius_squared);

    __m128i one  = _mm_set1_epi32(1), zero = _mm_setzero_si128();
    __m256d i2   = _mm256_mul_pd(ci,ci), r2 = _mm256_mul_pd(cr,cr);

    __m128i notescaped = to128(_mm256_and_pd(
                               _mm256_cmp_pd(
                                   _mm256_fmadd_pd(i2,
                                       _mm256_fmadd_pd(i2, _mm256_set1_pd(8),
                                                       _mm256_fmadd_pd(r2, _mm256_set1_pd(16), _mm256_set1_pd(-3))),
                                       _mm256_fmadd_pd(r2, _mm256_fmadd_pd(r2, _mm256_set1_pd(8), _mm256_set1_pd(-3)), cr)),
                                   _mm256_set1_pd(3./32), _CMP_GE_OQ),
                               _mm256_cmp_pd(
                                   _mm256_fmadd_pd(_mm256_add_pd(cr,_mm256_set1_pd(1)),_mm256_add_pd(cr,_mm256_set1_pd(1)), i2),
                                   _mm256_set1_pd(1./16), _CMP_GE_OQ)));

    __m128i iter = _mm_and_si128(_mm_set1_epi32(maxiter), notescaped);


    while(!_mm_testz_si128(notescaped, notescaped))
    {
        dist       = ifelse(notescaped, _mm256_fmadd_pd(cr,cr, i2), dist);

        notescaped = _mm_and_si128(notescaped,
                     _mm_andnot_si128(_mm_cmpeq_epi32(iter, zero),
                                      to128(_mm256_cmp_pd(dist, limit, _CMP_LT_OQ))));
        iter       = _mm_add_epi32(iter, notescaped);

        ci = _mm256_fmadd_pd(_mm256_mul_pd(cr, ci), _mm256_set1_pd(2), zi);
        cr = _mm256_fmadd_pd(cr,cr, _mm256_sub_pd(zr, i2));
        i2 = _mm256_mul_pd(ci,ci);

        if(WithMoment)
        {
            __m128i moment = _mm_cmpeq_epi32(_mm_and_si128(iter, _mm_sub_epi32(iter, one)), zero);

            iter = _mm_andnot_si128(to128(_mm256_and_pd(_mm256_cmp_pd(cr,sr,_CMP_EQ_OQ),
                                                        _mm256_cmp_pd(ci,si,_CMP_EQ_OQ))), iter);
            sr = ifelse(moment, cr, sr);
            si = ifelse(moment, ci, si);
        }
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
    bool NeedMoment = true;

    MAINLOOP_START(1);
    while(MAINLOOP_GET_CONDITION())
    {
        std::vector<unsigned> pixels (Xres * Yres);

        double zr, zi, xscale, yscale; MAINLOOP_SET_COORDINATES();

        #if defined(__AVX2__) || defined(__AVX512F__)
        constexpr unsigned N=8;
        #else
        constexpr unsigned N=4;
        #endif

        switch(N)
        {
            case 4:
            {
                unsigned n_zeroes = 0;
                #pragma omp parallel for schedule(dynamic) reduction(+:n_zeroes)
                for(unsigned y=0; y<Yres; ++y)
                {
                    __m128i num_zeroes = _mm_setzero_si128();
                    __m256d i = _mm256_set1_pd( zi+yscale*int(y-Yres/2) );

                    for(unsigned x=0; x<Xres/N*N; x += N)
                    {
                        __m256d r = _mm256_fmadd_pd(_mm256_set1_pd(xscale), _mm256_add_pd(_mm256_set_pd(3,2,1,0),
                                                                                          _mm256_set1_pd(int(x-Xres/2))),
                                                    _mm256_set1_pd(zr));

                        __m256d results = NeedMoment ? Iterate<true>(r,i) : Iterate<false>(r,i);

                        num_zeroes = _mm_sub_epi32(num_zeroes, to128(_mm256_cmp_pd(results, _mm256_setzero_pd(), _CMP_EQ_OQ)));

                        for(unsigned n=0; n<N; ++n) { pixels[y*Xres + x+n] = Color(x+n,y, results[n]); }
                    }
                    num_zeroes = _mm_add_epi32(num_zeroes, _mm_srli_si128(num_zeroes, 8)); // 0+2, 1+3, 2, 3
                    num_zeroes = _mm_hadd_epi32(num_zeroes, num_zeroes); // 0+2+1+3, ...
                    n_zeroes += _mm_extract_epi32(num_zeroes,0);
                }
                NeedMoment = n_zeroes >= int(Xres*Yres/1024);
                break;
            }

            case 8:
            {
                unsigned n_zeroes = 0;
                #pragma omp parallel for schedule(dynamic) reduction(+:n_zeroes)
                for(unsigned y=0; y<Yres; ++y)
                {
                    __m256i num_zeroes = _mm256_setzero_si256();
                    __m512d i = _mm512_set1_pd( zi+yscale*int(y-Yres/2) );

                    for(unsigned x=0; x<Xres/N*N; x += N)
                    {
                        __m512d r = _mm512_fmadd_pd(_mm512_set1_pd(xscale), _mm512_add_pd(_mm512_set_pd(7,6,5,4,3,2,1,0),
                                                                                          _mm512_set1_pd(int(x-Xres/2))),
                                                    _mm512_set1_pd(zr));

                        __m512d results = NeedMoment ? Iterate<true>(r,i) : Iterate<false>(r,i);

                        num_zeroes = _mm256_sub_epi32(num_zeroes, to256(_mm512_cmp_pd_mask(results, _mm512_setzero_pd(), _MM_CMPINT_EQ)));

                        for(unsigned n=0; n<N; ++n) { pixels[y*Xres + x+n] = Color(x+n,y, results[n]); }
                    }
                    __m128i z128 = _mm_add_epi32(extract128(num_zeroes,0), extract128(num_zeroes,1)); // 0+4, 1+5, 2+6, 3+7
                    z128 = _mm_add_epi32(z128, _mm_srli_si128(z128, 8)); // 0+4+2+6, 1+5+3+7, 2+6, 3+7
                    z128 = _mm_hadd_epi32(z128, z128); // 0+4+2+6+1+5+3+7, ...
                    n_zeroes += _mm_extract_epi32(z128,0);
                }
                NeedMoment = n_zeroes >= int(Xres*Yres/1024);
                break;
            }
        }

        MAINLOOP_PUT_RESULT(pixels);
    }
    MAINLOOP_FINISH();
}
