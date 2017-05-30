#include "common.inc"

#include <x86intrin.h>

#include "simd_emu.hh" // Optional: Emulate intrinsics when they are not available for this platform


__m128i to128(__m256d v)
    { __m256i c = _mm256_castpd_si256(v); return _mm_packs_epi16(extract128(c,0), extract128(c,1)); }
__m256d ifelse(__m256i c, __m256d ok, __m256d nok) { return _mm256_blendv_pd(nok, ok, _mm256_castsi256_pd(c)); }
__m256d ifelse(__m128i c, __m256d ok, __m256d nok) { return ifelse(_mm256_cvtepi32_epi64(c), ok, nok); }

__m256d _mm256_log2_pd(__m256d x) /* log2(x) for four positive doubles */
{
    //return _mm256_set_pd(std::log2(x[3]),std::log2(x[2]),std::log2(x[1]),std::log2(x[0]));

    __m256d half = _mm256_set1_pd(0.5);
    // x = frexp(x, &e);
    __m256i e = _mm256_srli_epi64(_mm256_castpd_si256(x), 52);
    __m256i m = _mm256_and_si256(_mm256_castpd_si256(x), _mm256_set1_epi64x((1ull << 52)-1));
    x = _mm256_or_pd(half, _mm256_castsi256_pd(m));

    __m256d ltid = _mm256_cmp_pd(x, _mm256_set1_pd(1/std::sqrt(2.)), _CMP_LT_OQ);
    __m256i lti = _mm256_castpd_si256(ltid);
    // if(lt) --e;
    __m256d dbl_e = _mm256_sub_pd(_mm256_cvtepi64_pd(_mm256_add_epi64(e,lti)), _mm256_set1_pd(1022));

    // if(lt) z = x-0.5; else z = x-1;
    // z = x - (lt ? 0.5 : 1);
    // z = x - 0.5 - (~lt & 0.5)
    __m256d z = _mm256_sub_pd(x, _mm256_add_pd(half, _mm256_andnot_pd(ltid, half)));
    // y = 0.5 * x + (lt ? -0.25 : 0) + 0.5
    // y = 0.5 * x + lt*0.25          + 0.5
    // y = 0.5 * (x + lt*0.5)         + 0.5
    // y = 0.5 * (x - (lt&0.5))       + 0.5
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
    // Multiply log of fraction by log2(e) and base 2 exponent by 1
    return _mm256_fmadd_pd(x, _mm256_mul_pd(y, _mm256_set1_pd(std::log2(std::exp(1.)))), dbl_e);
}

__m256d Iterate(__m256d zr, __m256d zi) __attribute__((noinline));
__m256d Iterate(__m256d zr, __m256d zi)
{
    const double escape_radius_squared = 6*6;
    const int maxiter = 8100;
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
        constexpr unsigned N=4;

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

        display.Put(pixels);
    }
    std::printf("\n%u frames rendered\n", display.frame);
}
