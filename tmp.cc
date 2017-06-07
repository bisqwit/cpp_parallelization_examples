#include <cmath>
#include <array>

template<std::size_t N>
std::array<double,N> plog2(const std::array<double, N>& value)
{
    constexpr int mantissa_bits = 52, exponent_bias = 1022;
    const double  half         = 0.5;
    std::uint64_t half_bits    = reinterpret_cast<const std::uint64_t&>(half);
    // x = frexp(x, &e);
    std::array<int,N> e, lt;
    std::array<uint64_t,N> m;
    std::array<double,N> x, dbl_e, z, y, u, t, result;
    for(unsigned n=0; n<N; ++n) { m[n] = reinterpret_cast<const std::uint64_t&>(value[n]);
                                  e[n] = m[n] >> mantissa_bits;
                                  m[n] &= std::uint64_t((1ull << mantissa_bits)-1);
                                  m[n] |= half_bits;
                                  x[n] = reinterpret_cast<const double&>(m[n]); }
    for(unsigned n=0; n<N; ++n) { lt[n] = (x[n] < 1/std::sqrt(2.)) ? -1 : 0;
                                  dbl_e[n] = e[n] + lt[n] - exponent_bias;
                                  z[n] = x[n] - (half + (lt[n] ? 0. : half));
                                  y[n] = half * (x[n] - (lt[n] ? half : 0.)) + half;
                                  x[n] = z[n]/y[n];
                                  z[n] = x[n]*x[n];
                                  u[n] = z[n]      + -3.56722798512324312549E1;
                                  t[n] =             -7.89580278884799154124E-1;
                                  u[n] = u[n]*z[n] +  3.12093766372244180303E2;
                                  t[n] = t[n]*z[n] +  1.63866645699558079767E1;
                                  u[n] = u[n]*z[n] + -7.69691943550460008604E2;
                                  t[n] = t[n]*z[n] + -6.41409952958715622951E1;
                                  y[n] = z[n]* (t[n]/u[n]) + (half+half);
                                  result[n] = x[n]*(y[n]*std::log2(std::exp(1.))) + dbl_e[n]; }
    return result;
}



template std::array<double,8> plog2(const std::array<double,8>& value);
