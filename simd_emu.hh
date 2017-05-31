#ifndef __AVX__
union my256 { struct { __m128d d[2]; }; struct { __m128i i[2]; };
              my256(__m128d dhi, __m128d dlo) { d[0] = dlo; d[1] = dhi; }
              my256(__m128i ihi, __m128i ilo) { i[0] = ilo; i[1] = ihi; }
              my256() = delete;
              double operator[](unsigned n) const { return d[n/2][n%2]; }
            };
#define __m256d my256
#define __m256i my256
#define _mm256_castpd_si256(x)         (x)
#define _mm256_castsi256_pd(x)         (x)
#define _mm256_castsi256_si128(x)      (x).i[0]
#define _mm256_castsi128_si256(x)      my256(_mm_setzero_si128(), x)
#define _mm256_extractf128_si256(x,n)  (x).i[n]
#define _mm256_and_pd(x,y)             ([](const my256& a, const my256& b) { return my256(_mm_and_pd(a.d[1],b.d[1]), _mm_and_pd(a.d[0],b.d[0])); })(x,y)
#define _mm256_or_pd(x,y)              ([](const my256& a, const my256& b) { return my256(_mm_or_pd( a.d[1],b.d[1]), _mm_or_pd( a.d[0],b.d[0])); })(x,y)
#define _mm256_add_pd(x,y)             ([](const my256& a, const my256& b) { return my256(_mm_add_pd(a.d[1],b.d[1]), _mm_add_pd(a.d[0],b.d[0])); })(x,y)
#define _mm256_sub_pd(x,y)             ([](const my256& a, const my256& b) { return my256(_mm_sub_pd(a.d[1],b.d[1]), _mm_sub_pd(a.d[0],b.d[0])); })(x,y)
#define _mm256_mul_pd(x,y)             ([](const my256& a, const my256& b) { return my256(_mm_mul_pd(a.d[1],b.d[1]), _mm_mul_pd(a.d[0],b.d[0])); })(x,y)
#define _mm256_div_pd(x,y)             ([](const my256& a, const my256& b) { return my256(_mm_div_pd(a.d[1],b.d[1]), _mm_div_pd(a.d[0],b.d[0])); })(x,y)
#define _mm256_andnot_pd(x,y)          ([](const my256& a, const my256& b) { return my256(_mm_andnot_pd(a.d[1],b.d[1]), _mm_andnot_pd(a.d[0],b.d[0])); })(x,y)
#define _mm256_cmp_pd_CMP_LT_OQ(x,y)   ([](const my256& a, const my256& b) { return my256(_mm_cmplt_pd(a.d[1],b.d[1]), _mm_cmplt_pd(a.d[0],b.d[0])); })(x,y)
#define _mm256_cmp_pd_CMP_EQ_OQ(x,y)   ([](const my256& a, const my256& b) { return my256(_mm_cmpeq_pd(a.d[1],b.d[1]), _mm_cmpeq_pd(a.d[0],b.d[0])); })(x,y)
#define _mm256_insertf128_si256(x,y,n) ([](my256 xx,__m128i yy,int nn){xx.i[nn] = yy; return xx;}(x,y,n))
#define _mm256_set1_pd(x)              ([](__m128d v){ return my256(v,v); })(_mm_set1_pd(x))
#define _mm256_set1_epi64x(x)          ([](__m128i v){ return my256(v,v); })(_mm_set1_epi64x(x))
#define _mm256_set_pd(d,c,b,a)         my256(_mm_set_pd(d,c),_mm_set_pd(b,a))
#define _mm256_cmp_pd(x,y,v)           _mm256_cmp_pd##v(x,y)
#define _mm256_cvtepi32_pd(x)          ([](__v4si v) { return _mm256_set_pd(v[3],v[2],v[1],v[0]); })((__v4si)(x))
#define _mm256_extract_epi64(x,n)      ([](const my256& x, unsigned nn) { return _mm_extract_epi64(x.i[nn/2], nn%2); })(x,n)
#endif

#ifndef __FMA__
 #define _mm256_fmadd_pd(a,b,c) _mm256_add_pd(_mm256_mul_pd(a,b), c)
#endif

#if !defined(__AVX512VL__) && !defined(__AVX512DQ__)
 #ifdef __AVX2__
  #define _mm256_cvtepi64_pd(d) _mm256_cvtepi32_pd(_mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_shuffle_epi32(d,232),232)))
 #else
  #define _mm256_cvtepi64_pd(d) ([](const __m256i& x){ return _mm256_set_pd(_mm256_extract_epi64(x,3),\
                                                                            _mm256_extract_epi64(x,2),\
                                                                            _mm256_extract_epi64(x,1),\
                                                                            _mm256_extract_epi64(x,0)); }(d))
 #endif
#endif

#define extract128(c,n)                (n ? _mm256_extractf128_si256(c, n) : _mm256_castsi256_si128(c))

#ifndef __AVX2__
 #define compose256(a,b)               _mm256_insertf128_si256(_mm256_castsi128_si256(a),b,1)
 #define _mm256_blendv_pd(nok,ok,cond) _mm256_or_pd(_mm256_and_pd(cond,ok), _mm256_andnot_pd(cond,nok))
 #define _mm256_cvtepi32_epi64(c)       compose256(_mm_cvtepi32_epi64(c), _mm_cvtepi32_epi64(_mm_shuffle_epi32(c,78)))
 #define _mm256_add_epi64(a,b) compose256(_mm_add_epi64(extract128(a,0), extract128(b,0)), \
                                          _mm_add_epi64(extract128(a,1), extract128(b,1)))
 #define _mm256_and_si256(a,b)         _mm256_castpd_si256(_mm256_and_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b)))
 #define _mm256_andnot_si256(a,b)      _mm256_castpd_si256(_mm256_andnot_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b)))
 #define _mm256_shuffle_epi32(x,n)     _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(x),_mm256_setzero_ps(), n))
 #define _mm256_srli_epi64(a,n) compose256(_mm_srli_epi64(extract128(a,0), n), \
                                           _mm_srli_epi64(extract128(a,1), n))
#endif

inline __m256d _mm256_log2_pd(__m256d x) /* log2(x) for four positive doubles */
{
    constexpr int mantissa_bits = 52, exponent_bias = 1022;
    //return _mm256_set_pd(std::log2(x[3]),std::log2(x[2]),std::log2(x[1]),std::log2(x[0]));

    __m256d half = _mm256_set1_pd(0.5);
    // x = frexp(x, &e);
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
    // Multiply log of fraction by log2(e) and base 2 exponent by 1
    return _mm256_fmadd_pd(x, _mm256_mul_pd(y, _mm256_set1_pd(std::log2(std::exp(1.)))), dbl_e);
}
