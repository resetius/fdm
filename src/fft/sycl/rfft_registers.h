#pragma once
// Real-input radix-2 FFT for SYCL. A length-N real sequence is represented by
// one length-N/2 complex transform. N is a template parameter so the compiler
// can unroll the butterflies and keep the working arrays in private storage.
//
// Packing matches the pFFT_1 convention used across this code base:
//   out[m]   = sum_i cos(2 pi i m / N) v[i],   m = 0..N/2
//   out[N-m] = sum_i sin(2 pi i m / N) v[i],   0 < m < N/2
//
// Addressing.  Element e of line t sits at base(t) + e*stride, with
// base(t) = (t/inner)*outer + t%inner.  That covers both directions of a
// [phi][z][r] array: phi uses inner=nr, outer=nr, stride=nz*nr; z uses
// inner=nr, outer=nz*nr, stride=nr.

#include <sycl/sycl.hpp>

#include <cmath>
#include <vector>

namespace fdm::fft_sycl {

constexpr int ilog2c(int n) { return n <= 1 ? 0 : 1 + ilog2c(n/2); }

constexpr bool is_power_of_two(int n) { return n > 0 && (n & (n-1)) == 0; }

template<int M>
constexpr int brev(int x) {
    int r = 0;
    for (int b = 0; b < ilog2c(M); ++b) { r |= ((x>>b)&1) << (ilog2c(M)-1-b); }
    return r;
}

// The table contains the length-M complex twiddles followed by the length-N
// real-transform twiddles W_N^m, m=0..M.
template<typename T>
std::vector<T> make_twiddles(int N) {
    const int M = N/2;
    std::vector<T> tw(2*M + 2*(M+1));
    for (int q = 0; q < M; ++q) {
        tw[2*q]   = T(std::cos(-2*M_PI*q/M));
        tw[2*q+1] = T(std::sin(-2*M_PI*q/M));
    }
    for (int m = 0; m <= M; ++m) {
        tw[2*M + 2*m]   = T(std::cos(-2*M_PI*m/N));
        tw[2*M + 2*m+1] = T(std::sin(-2*M_PI*m/N));
    }
    return tw;
}

namespace detail {

// In-place radix-2 decimation in time on registers; input already bit-reversed.
template<int M, typename T>
inline void butterflies(T (&re)[M], T (&im)[M], const T* tw) {
    constexpr int LM = ilog2c(M);
#pragma unroll
    for (int s = 1; s <= LM; ++s) {
        const int len = 1 << s, half = len >> 1, step = M/len;
#pragma unroll
        for (int blk = 0; blk < M; blk += len) {
#pragma unroll
            for (int u = 0; u < half; ++u) {
                const T wr = tw[2*(u*step)];
                const T wi = tw[2*(u*step)+1];
                const int a = blk+u, b = a+half;
                const T xr = re[b]*wr - im[b]*wi;
                const T xi = re[b]*wi + im[b]*wr;
                re[b] = re[a]-xr; im[b] = im[a]-xi;
                re[a] = re[a]+xr; im[a] = im[a]+xi;
            }
        }
    }
}

} // namespace detail

// out[m] and out[N-m] as described above, scaled by `scale`.
template<int N, typename T>
void real_forward(sycl::queue& q, T* out, const T* in, const T* tw, T scale,
                  int lines, int stride, int inner, int outer) {
    static_assert(is_power_of_two(N) && N >= 4, "N must be a power of two");
    constexpr int M = N/2;
    q.parallel_for(sycl::range<1>(size_t(lines)), [=](sycl::id<1> gid) {
        const int t = int(gid[0]);
        const int base = (t/inner)*outer + t%inner;

        T re[M], im[M];
        // z[p] = v[2p] + i v[2p+1], loaded bit-reversed for the in-place pass
#pragma unroll
        for (int p = 0; p < M; ++p) {
            const int b = brev<M>(p);
            re[p] = in[(2*b)*stride + base];
            im[p] = in[(2*b+1)*stride + base];
        }
        detail::butterflies<M>(re, im, tw);

        // X[m] = E[m] + W_N^m O[m], with E and O recovered from Z.
#pragma unroll
        for (int m = 0; m <= M; ++m) {
            const int mm = (m == M) ? 0 : m;
            const int mc = (M-m) % M;
            const T zr = re[mm], zi =  im[mm];
            const T cr = re[mc], ci = -im[mc];
            const T er = T(0.5)*(zr+cr), ei = T(0.5)*(zi+ci);
            const T dr = T(0.5)*(zr-cr), di = T(0.5)*(zi-ci);
            const T orr = di, ori = -dr;
            const T wr = tw[2*M + 2*m], wi = tw[2*M + 2*m+1];
            out[m*stride + base] = scale*(er + (orr*wr - ori*wi));
            if (m > 0 && m < M) {
                out[(N-m)*stride + base] = -scale*(ei + (orr*wi + ori*wr));
            }
        }
    });
}

// The mirror of real_forward: packed spectrum in, real line out.  Reconstructs
// E and O from the Hermitian spectrum, runs the length-M transform backwards,
// and interleaves the result.
template<int N, typename T>
void real_inverse(sycl::queue& q, T* out, const T* in, const T* tw, T scale,
                  int lines, int stride, int inner, int outer) {
    static_assert(is_power_of_two(N) && N >= 4, "N must be a power of two");
    constexpr int M = N/2;
    q.parallel_for(sycl::range<1>(size_t(lines)), [=](sycl::id<1> gid) {
        const int t = int(gid[0]);
        const int base = (t/inner)*outer + t%inner;

        T re[M], im[M];
#pragma unroll
        for (int p = 0; p < M; ++p) {
            const int b = brev<M>(p);
            // X[b] and conj(X[M-b]) from the packed halves
            const T xr  = in[b*stride + base];
            const T xi  = (b > 0) ? -in[(N-b)*stride + base] : T(0);
            // X[M-b], not X[(M-b) mod M]: at b = 0 that is the Nyquist term,
            // which is real and lives at index M.  The spectrum runs 0..M here,
            // unlike Z in the forward pass, which is M-periodic.
            const int c = M-b;
            const T cr  = in[c*stride + base];
            const T ci  = (c > 0 && c < M) ? -in[(N-c)*stride + base] : T(0);
            // conj(X[M-b])
            const T kr = cr, ki = -ci;
            const T er = T(0.5)*(xr+kr), ei = T(0.5)*(xi+ki);
            const T dr = T(0.5)*(xr-kr), di = T(0.5)*(xi-ki);
            // O = conj(W_N^b) * D, then Z = E + i O
            const T wr =  tw[2*M + 2*b];
            const T wi = -tw[2*M + 2*b+1];
            const T orr = dr*wr - di*wi;
            const T ori = dr*wi + di*wr;
            // conjugated input, so one forward pass performs the inverse
            re[p] =  er - ori;
            im[p] = -(ei + orr);
        }
        detail::butterflies<M>(re, im, tw);

#pragma unroll
        for (int p = 0; p < M; ++p) {
            out[(2*p)*stride + base]   =  scale*re[p];
            out[(2*p+1)*stride + base] = -scale*im[p];
        }
    });
}

} // namespace fdm::fft_sycl
