#pragma once

// AVX2 and AVX512 vectorized cyclic reduction (Kershaw general layout).
// Structure mirrors cyclic_reduction_neon.h.
// Three variants per ISA:
//   1. crkg   – full factorization + solve
//   2. crkgc  – solve only (d/e/f pre-factorized by crkg)
//   3. crkgPN – class with prepare() + execute() (precomputed alpha/gamma)
//
// NOT compiled on non-x86 targets; guarded by __AVX2__ / __AVX512F__.

#include <vector>
#include <cmath>
#include <type_traits>

#ifdef __AVX2__
#include <immintrin.h>

namespace fdm {

// ============================================================
// Deinterleave helpers
// ============================================================
//
// Kershaw layout: array has alternating "left-neighbour / node" pairs.
// Given two consecutive 256-bit loads lo = ptr[0..7] and hi = ptr[8..15]:
//   evens = {ptr[0], ptr[2], ptr[4], ptr[6], ptr[8], ptr[10], ptr[12], ptr[14]}
//   odds  = {ptr[1], ptr[3], ptr[5], ptr[7], ptr[9], ptr[11], ptr[13], ptr[15]}
//
// Step 1: _mm256_shuffle_ps with 0x88/0xDD extracts even/odd within each
//         128-bit lane, but interleaves the two sources:
//   e_shuf = [lo[0], lo[2], hi[0], hi[2],  lo[4], lo[6], hi[4], hi[6]]
// Step 2: _mm256_permutevar8x32_ps reorders to the correct final layout.

static inline void deinterleave_ps(__m256 lo, __m256 hi,
                                   __m256 &evens, __m256 &odds)
{
    const __m256i perm = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);
    evens = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(lo, hi, 0x88), perm);
    odds  = _mm256_permutevar8x32_ps(_mm256_shuffle_ps(lo, hi, 0xDD), perm);
}

// 4-wide double deinterleave.
// lo = ptr[0..3], hi = ptr[4..7]
// evens = {ptr[0], ptr[2], ptr[4], ptr[6]},  odds = {ptr[1], ptr[3], ptr[5], ptr[7]}
static inline void deinterleave_pd(__m256d lo, __m256d hi,
                                   __m256d &evens, __m256d &odds)
{
    // unpacklo/hi within 128-bit lanes, then permute 64-bit lanes with 0xD8
    evens = _mm256_permute4x64_pd(_mm256_unpacklo_pd(lo, hi), 0xD8);
    odds  = _mm256_permute4x64_pd(_mm256_unpackhi_pd(lo, hi), 0xD8);
}

// ============================================================
// Variant 1 – full crkg (AVX2, float, 8-wide)
// ============================================================
inline void cyclic_reduction_kershaw_general_avx2(
    float *__restrict d, float *__restrict e, float *__restrict f,
    float *__restrict b, int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;
    off    = 0;
    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;
        j   = off + 1;
        dst = off + n_curr;

        // 8-wide SIMD: processes j,j+2,...,j+14 simultaneously (j advances 16)
        for (; j + 15 < off + n_curr; j += 16, dst += 8) {
            __m256 d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m256 d_right, e_right, f_right, b_right, dummy;

            deinterleave_ps(_mm256_loadu_ps(d+j-1), _mm256_loadu_ps(d+j+7),  d_left, d_node);
            deinterleave_ps(_mm256_loadu_ps(e+j-1), _mm256_loadu_ps(e+j+7),  e_left, e_node);
            deinterleave_ps(_mm256_loadu_ps(f+j-1), _mm256_loadu_ps(f+j+7),  f_left, f_node);
            deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7),  b_left, b_node);

            deinterleave_ps(_mm256_loadu_ps(d+j+1), _mm256_loadu_ps(d+j+9),  d_right, dummy);
            deinterleave_ps(_mm256_loadu_ps(e+j+1), _mm256_loadu_ps(e+j+9),  e_right, dummy);
            deinterleave_ps(_mm256_loadu_ps(f+j+1), _mm256_loadu_ps(f+j+9),  f_right, dummy);
            deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9),  b_right, dummy);

            __m256 zero  = _mm256_setzero_ps();
            __m256 alpha   = _mm256_div_ps(_mm256_sub_ps(zero, e_node), d_left);
            __m256 gamma_v = _mm256_div_ps(_mm256_sub_ps(zero, f_node), d_right);

            _mm256_storeu_ps(d+dst, _mm256_fmadd_ps(gamma_v, e_right,
                                    _mm256_fmadd_ps(alpha,   f_left,  d_node)));
            _mm256_storeu_ps(b+dst, _mm256_fmadd_ps(gamma_v, b_right,
                                    _mm256_fmadd_ps(alpha,   b_left,  b_node)));
            _mm256_storeu_ps(e+dst, _mm256_mul_ps(alpha,   e_left));
            _mm256_storeu_ps(f+dst, _mm256_mul_ps(gamma_v, f_right));
        }

        for (; j + 1 < off + n_curr; j += 2, dst++) {
            float a = -e[j]/d[j-1], g = -f[j]/d[j+1];
            d[dst] = d[j] + a*f[j-1] + g*e[j+1];
            b[dst] = b[j] + a*b[j-1] + g*b[j+1];
            e[dst] = a*e[j-1]; f[dst] = g*f[j+1];
        }
        for (; j < off + n_curr; j += 2, dst++) {
            float a = -e[j]/d[j-1];
            d[dst] = d[j] + a*f[j-1]; b[dst] = b[j] + a*b[j-1]; e[dst] = a*e[j-1];
        }

        off += n_curr; n_curr = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr*2+1 : n_curr*2;

        for (j = off, dst = off-n_next+1; j < off+n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j]*b[j+1]) / d[j];
        j += 2;

        // 8-wide back-sub: even positions j,j+2,...,j+14 are independent
        for (; j + 15 < off; j += 16) {
            __m256 b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;

            deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7),  b_left, b_node);
            deinterleave_ps(_mm256_loadu_ps(d+j-1), _mm256_loadu_ps(d+j+7),  d_left, d_node);
            deinterleave_ps(_mm256_loadu_ps(e+j-1), _mm256_loadu_ps(e+j+7),  e_left, e_node);
            deinterleave_ps(_mm256_loadu_ps(f+j-1), _mm256_loadu_ps(f+j+7),  f_left, f_node);

            __m256 b_right, rdum;
            deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9),  b_right, rdum);

            // res = (b_node - e_node*b_left - f_node*b_right) / d_node
            __m256 res = _mm256_div_ps(
                _mm256_fnmadd_ps(f_node, b_right,
                _mm256_fnmadd_ps(e_node, b_left, b_node)),
                d_node);

            // scatter to stride-2 positions
            float tmp[8]; _mm256_storeu_ps(tmp, res);
            for (int k = 0; k < 8; k++) b[j + 2*k] = tmp[k];
        }

        for (; j + 1 < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1] - f[j]*b[j+1]) / d[j];
        for (; j < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1]) / d[j];

        off -= n_next; n_curr = n_next;
    }
}

// ---- double overload (4-wide) ----
inline void cyclic_reduction_kershaw_general_avx2(
    double *__restrict d, double *__restrict e, double *__restrict f,
    double *__restrict b, int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;
    off    = 0;
    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;
        j   = off + 1;
        dst = off + n_curr;

        // 4-wide: j,j+2,j+4,j+6 (j advances 8)
        for (; j + 7 < off + n_curr; j += 8, dst += 4) {
            __m256d d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m256d d_right, e_right, f_right, b_right, dummy;

            deinterleave_pd(_mm256_loadu_pd(d+j-1), _mm256_loadu_pd(d+j+3),  d_left, d_node);
            deinterleave_pd(_mm256_loadu_pd(e+j-1), _mm256_loadu_pd(e+j+3),  e_left, e_node);
            deinterleave_pd(_mm256_loadu_pd(f+j-1), _mm256_loadu_pd(f+j+3),  f_left, f_node);
            deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3),  b_left, b_node);

            deinterleave_pd(_mm256_loadu_pd(d+j+1), _mm256_loadu_pd(d+j+5),  d_right, dummy);
            deinterleave_pd(_mm256_loadu_pd(e+j+1), _mm256_loadu_pd(e+j+5),  e_right, dummy);
            deinterleave_pd(_mm256_loadu_pd(f+j+1), _mm256_loadu_pd(f+j+5),  f_right, dummy);
            deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5),  b_right, dummy);

            __m256d zero  = _mm256_setzero_pd();
            __m256d alpha   = _mm256_div_pd(_mm256_sub_pd(zero, e_node), d_left);
            __m256d gamma_v = _mm256_div_pd(_mm256_sub_pd(zero, f_node), d_right);

            _mm256_storeu_pd(d+dst, _mm256_fmadd_pd(gamma_v, e_right,
                                    _mm256_fmadd_pd(alpha,   f_left,  d_node)));
            _mm256_storeu_pd(b+dst, _mm256_fmadd_pd(gamma_v, b_right,
                                    _mm256_fmadd_pd(alpha,   b_left,  b_node)));
            _mm256_storeu_pd(e+dst, _mm256_mul_pd(alpha,   e_left));
            _mm256_storeu_pd(f+dst, _mm256_mul_pd(gamma_v, f_right));
        }

        for (; j + 1 < off + n_curr; j += 2, dst++) {
            double a = -e[j]/d[j-1], g = -f[j]/d[j+1];
            d[dst] = d[j] + a*f[j-1] + g*e[j+1];
            b[dst] = b[j] + a*b[j-1] + g*b[j+1];
            e[dst] = a*e[j-1]; f[dst] = g*f[j+1];
        }
        for (; j < off + n_curr; j += 2, dst++) {
            double a = -e[j]/d[j-1];
            d[dst] = d[j] + a*f[j-1]; b[dst] = b[j] + a*b[j-1]; e[dst] = a*e[j-1];
        }

        off += n_curr; n_curr = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr*2+1 : n_curr*2;

        for (j = off, dst = off-n_next+1; j < off+n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j]*b[j+1]) / d[j];
        j += 2;

        for (; j + 7 < off; j += 8) {
            __m256d b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;

            deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3),  b_left, b_node);
            deinterleave_pd(_mm256_loadu_pd(d+j-1), _mm256_loadu_pd(d+j+3),  d_left, d_node);
            deinterleave_pd(_mm256_loadu_pd(e+j-1), _mm256_loadu_pd(e+j+3),  e_left, e_node);
            deinterleave_pd(_mm256_loadu_pd(f+j-1), _mm256_loadu_pd(f+j+3),  f_left, f_node);

            __m256d b_right, rdum;
            deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5),  b_right, rdum);

            __m256d res = _mm256_div_pd(
                _mm256_fnmadd_pd(f_node, b_right,
                _mm256_fnmadd_pd(e_node, b_left, b_node)),
                d_node);

            double tmp[4]; _mm256_storeu_pd(tmp, res);
            for (int k = 0; k < 4; k++) b[j + 2*k] = tmp[k];
        }

        for (; j + 1 < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1] - f[j]*b[j+1]) / d[j];
        for (; j < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1]) / d[j];

        off -= n_next; n_curr = n_next;
    }
}

// ============================================================
// Variant 2 – _continue / crkgc (AVX2)
// d/e/f already factorized; only b is updated in forward sweep.
// ============================================================
inline void cyclic_reduction_kershaw_general_continue_avx2(
    float *__restrict d, float *__restrict e, float *__restrict f,
    float *__restrict b, int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;
    off    = 0;
    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;
        j   = off + 1;
        dst = off + n_curr;

        for (; j + 15 < off + n_curr; j += 16, dst += 8) {
            __m256 d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m256 d_right, b_right, dummy;

            deinterleave_ps(_mm256_loadu_ps(d+j-1), _mm256_loadu_ps(d+j+7),  d_left, d_node);
            deinterleave_ps(_mm256_loadu_ps(e+j-1), _mm256_loadu_ps(e+j+7),  e_left, e_node);
            deinterleave_ps(_mm256_loadu_ps(f+j-1), _mm256_loadu_ps(f+j+7),  f_left, f_node);
            deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7),  b_left, b_node);

            deinterleave_ps(_mm256_loadu_ps(d+j+1), _mm256_loadu_ps(d+j+9),  d_right, dummy);
            deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9),  b_right, dummy);

            __m256 zero  = _mm256_setzero_ps();
            __m256 alpha   = _mm256_div_ps(_mm256_sub_ps(zero, e_node), d_left);
            __m256 gamma_v = _mm256_div_ps(_mm256_sub_ps(zero, f_node), d_right);

            _mm256_storeu_ps(b+dst, _mm256_fmadd_ps(gamma_v, b_right,
                                    _mm256_fmadd_ps(alpha,   b_left,  b_node)));
        }

        for (; j + 1 < off + n_curr; j += 2, dst++) {
            float a = -e[j]/d[j-1], g = -f[j]/d[j+1];
            b[dst] = b[j] + a*b[j-1] + g*b[j+1];
        }
        for (; j < off + n_curr; j += 2, dst++) {
            float a = -e[j]/d[j-1];
            b[dst] = b[j] + a*b[j-1];
        }

        off += n_curr; n_curr = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr*2+1 : n_curr*2;

        for (j = off, dst = off-n_next+1; j < off+n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j]*b[j+1]) / d[j];
        j += 2;

        for (; j + 15 < off; j += 16) {
            __m256 b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;

            deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7),  b_left, b_node);
            deinterleave_ps(_mm256_loadu_ps(d+j-1), _mm256_loadu_ps(d+j+7),  d_left, d_node);
            deinterleave_ps(_mm256_loadu_ps(e+j-1), _mm256_loadu_ps(e+j+7),  e_left, e_node);
            deinterleave_ps(_mm256_loadu_ps(f+j-1), _mm256_loadu_ps(f+j+7),  f_left, f_node);

            __m256 b_right, rdum;
            deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9),  b_right, rdum);

            __m256 res = _mm256_div_ps(
                _mm256_fnmadd_ps(f_node, b_right,
                _mm256_fnmadd_ps(e_node, b_left, b_node)),
                d_node);

            float tmp[8]; _mm256_storeu_ps(tmp, res);
            for (int k = 0; k < 8; k++) b[j + 2*k] = tmp[k];
        }

        for (; j + 1 < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1] - f[j]*b[j+1]) / d[j];
        for (; j < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1]) / d[j];

        off -= n_next; n_curr = n_next;
    }
}

// ---- double overload ----
inline void cyclic_reduction_kershaw_general_continue_avx2(
    double *__restrict d, double *__restrict e, double *__restrict f,
    double *__restrict b, int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;
    off    = 0;
    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;
        j   = off + 1;
        dst = off + n_curr;

        for (; j + 7 < off + n_curr; j += 8, dst += 4) {
            __m256d d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m256d d_right, b_right, dummy;

            deinterleave_pd(_mm256_loadu_pd(d+j-1), _mm256_loadu_pd(d+j+3),  d_left, d_node);
            deinterleave_pd(_mm256_loadu_pd(e+j-1), _mm256_loadu_pd(e+j+3),  e_left, e_node);
            deinterleave_pd(_mm256_loadu_pd(f+j-1), _mm256_loadu_pd(f+j+3),  f_left, f_node);
            deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3),  b_left, b_node);

            deinterleave_pd(_mm256_loadu_pd(d+j+1), _mm256_loadu_pd(d+j+5),  d_right, dummy);
            deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5),  b_right, dummy);

            __m256d zero  = _mm256_setzero_pd();
            __m256d alpha   = _mm256_div_pd(_mm256_sub_pd(zero, e_node), d_left);
            __m256d gamma_v = _mm256_div_pd(_mm256_sub_pd(zero, f_node), d_right);

            _mm256_storeu_pd(b+dst, _mm256_fmadd_pd(gamma_v, b_right,
                                    _mm256_fmadd_pd(alpha,   b_left,  b_node)));
        }

        for (; j + 1 < off + n_curr; j += 2, dst++) {
            double a = -e[j]/d[j-1], g = -f[j]/d[j+1];
            b[dst] = b[j] + a*b[j-1] + g*b[j+1];
        }
        for (; j < off + n_curr; j += 2, dst++) {
            double a = -e[j]/d[j-1];
            b[dst] = b[j] + a*b[j-1];
        }

        off += n_curr; n_curr = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr*2+1 : n_curr*2;

        for (j = off, dst = off-n_next+1; j < off+n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j]*b[j+1]) / d[j];
        j += 2;

        for (; j + 7 < off; j += 8) {
            __m256d b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;

            deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3),  b_left, b_node);
            deinterleave_pd(_mm256_loadu_pd(d+j-1), _mm256_loadu_pd(d+j+3),  d_left, d_node);
            deinterleave_pd(_mm256_loadu_pd(e+j-1), _mm256_loadu_pd(e+j+3),  e_left, e_node);
            deinterleave_pd(_mm256_loadu_pd(f+j-1), _mm256_loadu_pd(f+j+3),  f_left, f_node);

            __m256d b_right, rdum;
            deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5),  b_right, rdum);

            __m256d res = _mm256_div_pd(
                _mm256_fnmadd_pd(f_node, b_right,
                _mm256_fnmadd_pd(e_node, b_left, b_node)),
                d_node);

            double tmp[4]; _mm256_storeu_pd(tmp, res);
            for (int k = 0; k < 4; k++) b[j + 2*k] = tmp[k];
        }

        for (; j + 1 < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1] - f[j]*b[j+1]) / d[j];
        for (; j < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1]) / d[j];

        off -= n_next; n_curr = n_next;
    }
}

// ============================================================
// Variant 3 – CyclicReductionAVX2<T>  (precomputed alpha/gamma)
// ============================================================
// prepare() is pure scalar (same as CyclicReductionNeon).
// execute() uses AVX2 SIMD:
//   forward: vld1 (contiguous alpha/gamma) + deinterleave b → fmadd → storeu
//   back-sub: deinterleave inv_d/e_bs/f_bs + deinterleave b → scatter

template<typename T>
class CyclicReductionAVX2 {
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>,
                  "CyclicReductionAVX2 requires float or double");
public:
    explicit CyclicReductionAVX2(int n)
        : n_(n)
        , q_((int)std::ceil(std::log2((double)(n+1))))
        , inv_d_mid_(T(0))
        , alpha_fwd_(2*n+2, T(0)), gamma_fwd_(2*n+2, T(0))
        , inv_d_bs_(2*n+2, T(0)), e_bs_(2*n+2, T(0)), f_bs_(2*n+2, T(0))
    {}

    void prepare(const T *d_in, const T *e_in, const T *f_in) {
        const int n = n_, q = q_;
        std::vector<T> dk(2*n+2, T(0)), ek(2*n+2, T(0)), fk(2*n+2, T(0));
        for (int i = 0; i < n; i++) { dk[i]=d_in[i]; ek[i]=e_in[i]; fk[i]=f_in[i]; }

        int j, l, n_curr = n, n_next, off = 0, dst;

        for (l = 1; l < q; l++) {
            n_next = (n_curr - n_curr % 2) / 2;
            j = off+1; dst = off+n_curr;
            for (; j+1 < off+n_curr; j += 2, dst++) {
                T a = -ek[j]/dk[j-1], g = -fk[j]/dk[j+1];
                alpha_fwd_[dst] = a; gamma_fwd_[dst] = g;
                dk[dst] = dk[j] + a*fk[j-1] + g*ek[j+1];
                ek[dst] = a*ek[j-1]; fk[dst] = g*fk[j+1];
            }
            for (; j < off+n_curr; j += 2, dst++) {
                T a = -ek[j]/dk[j-1];
                alpha_fwd_[dst] = a; gamma_fwd_[dst] = T(0);
                dk[dst] = dk[j] + a*fk[j-1]; ek[dst] = a*ek[j-1]; fk[dst] = T(0);
            }
            off += n_curr; n_curr = n_next;
        }
        inv_d_mid_ = T(1) / dk[off];
        n_curr = 1;

        for (int mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
            n_next = (n & mask) ? n_curr*2+1 : n_curr*2;
            j = off - n_next;
            inv_d_bs_[j] = T(1)/dk[j]; e_bs_[j] = T(0); f_bs_[j] = fk[j]/dk[j];
            j += 2;
            for (; j+1 < off; j += 2) {
                inv_d_bs_[j] = T(1)/dk[j]; e_bs_[j] = ek[j]/dk[j]; f_bs_[j] = fk[j]/dk[j];
            }
            for (; j < off; j += 2) {
                inv_d_bs_[j] = T(1)/dk[j]; e_bs_[j] = ek[j]/dk[j]; f_bs_[j] = T(0);
            }
            off -= n_next; n_curr = n_next;
        }
    }

    void execute(T *b) {
        const int n = n_, q = q_;
        int j, l, n_curr = n, n_next, off = 0, dst, mask;

        for (l = 1; l < q; l++) {
            n_next = (n_curr - n_curr % 2) / 2;
            j = off+1; dst = off+n_curr;

            if constexpr (std::is_same_v<T, float>) {
                for (; j + 15 < off + n_curr; j += 16, dst += 8) {
                    __m256 alpha_v = _mm256_loadu_ps(alpha_fwd_.data() + dst);
                    __m256 gamma_v = _mm256_loadu_ps(gamma_fwd_.data() + dst);
                    __m256 b_left, b_node;
                    deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7), b_left, b_node);
                    __m256 b_right, dummy;
                    deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9), b_right, dummy);
                    _mm256_storeu_ps(b+dst, _mm256_fmadd_ps(gamma_v, b_right,
                                            _mm256_fmadd_ps(alpha_v, b_left, b_node)));
                }
            } else {
                for (; j + 7 < off + n_curr; j += 8, dst += 4) {
                    __m256d alpha_v = _mm256_loadu_pd(alpha_fwd_.data() + dst);
                    __m256d gamma_v = _mm256_loadu_pd(gamma_fwd_.data() + dst);
                    __m256d b_left, b_node;
                    deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3), b_left, b_node);
                    __m256d b_right, dummy;
                    deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5), b_right, dummy);
                    _mm256_storeu_pd(b+dst, _mm256_fmadd_pd(gamma_v, b_right,
                                            _mm256_fmadd_pd(alpha_v, b_left, b_node)));
                }
            }

            for (; j+1 < off+n_curr; j += 2, dst++)
                b[dst] = b[j] + alpha_fwd_[dst]*b[j-1] + gamma_fwd_[dst]*b[j+1];
            for (; j < off+n_curr; j += 2, dst++)
                b[dst] = b[j] + alpha_fwd_[dst]*b[j-1];

            off += n_curr; n_curr = n_next;
        }

        b[off] = b[off] * inv_d_mid_;
        n_curr = 1;

        for (mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
            n_next = (n & mask) ? n_curr*2+1 : n_curr*2;

            for (j = off, dst = off-n_next+1; j < off+n_curr; j++, dst += 2)
                b[dst] = b[j];

            j = off - n_next;
            b[j] = inv_d_bs_[j]*b[j] - f_bs_[j]*b[j+1];
            j += 2;

            if constexpr (std::is_same_v<T, float>) {
                for (; j + 15 < off; j += 16) {
                    __m256 b_left, b_node, b_right, dummy;
                    deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7), b_left, b_node);
                    deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9), b_right, dummy);
                    // load precomputed at stride-2 positions via deinterleave (take odds lane)
                    __m256 inv_d_v, e_v, f_v, junk;
                    deinterleave_ps(_mm256_loadu_ps(inv_d_bs_.data()+j-1),
                                    _mm256_loadu_ps(inv_d_bs_.data()+j+7), junk, inv_d_v);
                    deinterleave_ps(_mm256_loadu_ps(e_bs_.data()+j-1),
                                    _mm256_loadu_ps(e_bs_.data()+j+7),     junk, e_v);
                    deinterleave_ps(_mm256_loadu_ps(f_bs_.data()+j-1),
                                    _mm256_loadu_ps(f_bs_.data()+j+7),     junk, f_v);
                    __m256 res = _mm256_fnmadd_ps(f_v, b_right,
                                 _mm256_fnmadd_ps(e_v, b_left,
                                 _mm256_mul_ps(inv_d_v, b_node)));
                    float tmp[8]; _mm256_storeu_ps(tmp, res);
                    for (int k = 0; k < 8; k++) b[j + 2*k] = tmp[k];
                }
            } else {
                for (; j + 7 < off; j += 8) {
                    __m256d b_left, b_node, b_right, dummy;
                    deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3), b_left, b_node);
                    deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5), b_right, dummy);
                    __m256d inv_d_v, e_v, f_v, junk;
                    deinterleave_pd(_mm256_loadu_pd(inv_d_bs_.data()+j-1),
                                    _mm256_loadu_pd(inv_d_bs_.data()+j+3), junk, inv_d_v);
                    deinterleave_pd(_mm256_loadu_pd(e_bs_.data()+j-1),
                                    _mm256_loadu_pd(e_bs_.data()+j+3),     junk, e_v);
                    deinterleave_pd(_mm256_loadu_pd(f_bs_.data()+j-1),
                                    _mm256_loadu_pd(f_bs_.data()+j+3),     junk, f_v);
                    __m256d res = _mm256_fnmadd_pd(f_v, b_right,
                                  _mm256_fnmadd_pd(e_v, b_left,
                                  _mm256_mul_pd(inv_d_v, b_node)));
                    double tmp[4]; _mm256_storeu_pd(tmp, res);
                    for (int k = 0; k < 4; k++) b[j + 2*k] = tmp[k];
                }
            }

            for (; j+1 < off; j += 2)
                b[j] = inv_d_bs_[j]*b[j] - e_bs_[j]*b[j-1] - f_bs_[j]*b[j+1];
            for (; j < off; j += 2)
                b[j] = inv_d_bs_[j]*b[j] - e_bs_[j]*b[j-1];

            off -= n_next; n_curr = n_next;
        }
    }

private:
    int n_, q_;
    T   inv_d_mid_;
    std::vector<T> alpha_fwd_, gamma_fwd_;
    std::vector<T> inv_d_bs_, e_bs_, f_bs_;
};

// ============================================================
// AVX-512 variants
// ============================================================
#ifdef __AVX512F__

// Deinterleave for 512-bit registers.
// Float: lo=ptr[0..15], hi=ptr[16..31]
//   evens = ptr[0,2,4,...,30], odds = ptr[1,3,...,31]
// Permutation after shuffle: [0,1,4,5,8,9,12,13, 2,3,6,7,10,11,14,15]
static inline void deinterleave_ps512(__m512 lo, __m512 hi,
                                      __m512 &evens, __m512 &odds)
{
    // shuffle_ps within each 128-bit lane (same as AVX2 but 4 lanes):
    // e_shuf = [lo[0],lo[2],hi[0],hi[2], lo[4],lo[6],hi[4],hi[6],
    //           lo[8],lo[10],hi[8],hi[10], lo[12],lo[14],hi[12],hi[14]]
    const __m512i perm = _mm512_set_epi32(
        15,14,11,10, 7,6,3,2, 13,12,9,8, 5,4,1,0);
    evens = _mm512_permutexvar_ps(perm, _mm512_shuffle_ps(lo, hi, 0x88));
    odds  = _mm512_permutexvar_ps(perm, _mm512_shuffle_ps(lo, hi, 0xDD));
}

// Double: lo=ptr[0..7], hi=ptr[8..15]
//   evens = ptr[0,2,4,6,8,10,12,14], odds = ptr[1,3,...,15]
// Permutation after unpacklo/hi: [0,2,4,6,1,3,5,7]
static inline void deinterleave_pd512(__m512d lo, __m512d hi,
                                      __m512d &evens, __m512d &odds)
{
    const __m512i perm = _mm512_set_epi64(7,5,3,1, 6,4,2,0);
    evens = _mm512_permutexvar_pd(perm, _mm512_unpacklo_pd(lo, hi));
    odds  = _mm512_permutexvar_pd(perm, _mm512_unpackhi_pd(lo, hi));
}

// ---- crkg AVX512 float (16-wide) ----
inline void cyclic_reduction_kershaw_general_avx512(
    float *__restrict d, float *__restrict e, float *__restrict f,
    float *__restrict b, int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;
    off    = 0;
    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;
        j   = off + 1;
        dst = off + n_curr;

        // 16-wide: j,j+2,...,j+30 (j advances 32)
        for (; j + 31 < off + n_curr; j += 32, dst += 16) {
            __m512 d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m512 d_right, e_right, f_right, b_right, dummy;

            deinterleave_ps512(_mm512_loadu_ps(d+j-1), _mm512_loadu_ps(d+j+15), d_left, d_node);
            deinterleave_ps512(_mm512_loadu_ps(e+j-1), _mm512_loadu_ps(e+j+15), e_left, e_node);
            deinterleave_ps512(_mm512_loadu_ps(f+j-1), _mm512_loadu_ps(f+j+15), f_left, f_node);
            deinterleave_ps512(_mm512_loadu_ps(b+j-1), _mm512_loadu_ps(b+j+15), b_left, b_node);

            deinterleave_ps512(_mm512_loadu_ps(d+j+1), _mm512_loadu_ps(d+j+17), d_right, dummy);
            deinterleave_ps512(_mm512_loadu_ps(e+j+1), _mm512_loadu_ps(e+j+17), e_right, dummy);
            deinterleave_ps512(_mm512_loadu_ps(f+j+1), _mm512_loadu_ps(f+j+17), f_right, dummy);
            deinterleave_ps512(_mm512_loadu_ps(b+j+1), _mm512_loadu_ps(b+j+17), b_right, dummy);

            __m512 alpha   = _mm512_div_ps(_mm512_sub_ps(_mm512_setzero_ps(), e_node), d_left);
            __m512 gamma_v = _mm512_div_ps(_mm512_sub_ps(_mm512_setzero_ps(), f_node), d_right);

            _mm512_storeu_ps(d+dst, _mm512_fmadd_ps(gamma_v, e_right,
                                    _mm512_fmadd_ps(alpha,   f_left,  d_node)));
            _mm512_storeu_ps(b+dst, _mm512_fmadd_ps(gamma_v, b_right,
                                    _mm512_fmadd_ps(alpha,   b_left,  b_node)));
            _mm512_storeu_ps(e+dst, _mm512_mul_ps(alpha,   e_left));
            _mm512_storeu_ps(f+dst, _mm512_mul_ps(gamma_v, f_right));
        }

        // fall through to AVX2 8-wide tail
        for (; j + 15 < off + n_curr; j += 16, dst += 8) {
            __m256 d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m256 d_right, e_right, f_right, b_right, dummy;
            deinterleave_ps(_mm256_loadu_ps(d+j-1), _mm256_loadu_ps(d+j+7),  d_left, d_node);
            deinterleave_ps(_mm256_loadu_ps(e+j-1), _mm256_loadu_ps(e+j+7),  e_left, e_node);
            deinterleave_ps(_mm256_loadu_ps(f+j-1), _mm256_loadu_ps(f+j+7),  f_left, f_node);
            deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7),  b_left, b_node);
            deinterleave_ps(_mm256_loadu_ps(d+j+1), _mm256_loadu_ps(d+j+9),  d_right, dummy);
            deinterleave_ps(_mm256_loadu_ps(e+j+1), _mm256_loadu_ps(e+j+9),  e_right, dummy);
            deinterleave_ps(_mm256_loadu_ps(f+j+1), _mm256_loadu_ps(f+j+9),  f_right, dummy);
            deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9),  b_right, dummy);
            __m256 zero = _mm256_setzero_ps();
            __m256 alpha   = _mm256_div_ps(_mm256_sub_ps(zero, e_node), d_left);
            __m256 gamma_v = _mm256_div_ps(_mm256_sub_ps(zero, f_node), d_right);
            _mm256_storeu_ps(d+dst, _mm256_fmadd_ps(gamma_v, e_right, _mm256_fmadd_ps(alpha, f_left, d_node)));
            _mm256_storeu_ps(b+dst, _mm256_fmadd_ps(gamma_v, b_right, _mm256_fmadd_ps(alpha, b_left, b_node)));
            _mm256_storeu_ps(e+dst, _mm256_mul_ps(alpha, e_left));
            _mm256_storeu_ps(f+dst, _mm256_mul_ps(gamma_v, f_right));
        }

        for (; j+1 < off+n_curr; j += 2, dst++) {
            float a=-e[j]/d[j-1], g=-f[j]/d[j+1];
            d[dst]=d[j]+a*f[j-1]+g*e[j+1]; b[dst]=b[j]+a*b[j-1]+g*b[j+1];
            e[dst]=a*e[j-1]; f[dst]=g*f[j+1];
        }
        for (; j < off+n_curr; j += 2, dst++) {
            float a=-e[j]/d[j-1];
            d[dst]=d[j]+a*f[j-1]; b[dst]=b[j]+a*b[j-1]; e[dst]=a*e[j-1];
        }
        off += n_curr; n_curr = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr*2+1 : n_curr*2;

        for (j = off, dst = off-n_next+1; j < off+n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j]*b[j+1]) / d[j];
        j += 2;

        for (; j + 31 < off; j += 32) {
            __m512 b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;
            deinterleave_ps512(_mm512_loadu_ps(b+j-1), _mm512_loadu_ps(b+j+15), b_left, b_node);
            deinterleave_ps512(_mm512_loadu_ps(d+j-1), _mm512_loadu_ps(d+j+15), d_left, d_node);
            deinterleave_ps512(_mm512_loadu_ps(e+j-1), _mm512_loadu_ps(e+j+15), e_left, e_node);
            deinterleave_ps512(_mm512_loadu_ps(f+j-1), _mm512_loadu_ps(f+j+15), f_left, f_node);
            __m512 b_right, rdum;
            deinterleave_ps512(_mm512_loadu_ps(b+j+1), _mm512_loadu_ps(b+j+17), b_right, rdum);
            __m512 res = _mm512_div_ps(
                _mm512_fnmadd_ps(f_node, b_right,
                _mm512_fnmadd_ps(e_node, b_left, b_node)), d_node);
            float tmp[16]; _mm512_storeu_ps(tmp, res);
            for (int k = 0; k < 16; k++) b[j + 2*k] = tmp[k];
        }
        for (; j + 15 < off; j += 16) {
            __m256 b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;
            deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7), b_left, b_node);
            deinterleave_ps(_mm256_loadu_ps(d+j-1), _mm256_loadu_ps(d+j+7), d_left, d_node);
            deinterleave_ps(_mm256_loadu_ps(e+j-1), _mm256_loadu_ps(e+j+7), e_left, e_node);
            deinterleave_ps(_mm256_loadu_ps(f+j-1), _mm256_loadu_ps(f+j+7), f_left, f_node);
            __m256 b_right, rdum;
            deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9), b_right, rdum);
            __m256 res = _mm256_div_ps(
                _mm256_fnmadd_ps(f_node, b_right,
                _mm256_fnmadd_ps(e_node, b_left, b_node)), d_node);
            float tmp[8]; _mm256_storeu_ps(tmp, res);
            for (int k = 0; k < 8; k++) b[j + 2*k] = tmp[k];
        }
        for (; j+1 < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1] - f[j]*b[j+1]) / d[j];
        for (; j < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1]) / d[j];

        off -= n_next; n_curr = n_next;
    }
}

// ---- crkg AVX512 double (8-wide) ----
inline void cyclic_reduction_kershaw_general_avx512(
    double *__restrict d, double *__restrict e, double *__restrict f,
    double *__restrict b, int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;
    off    = 0;
    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;
        j   = off + 1;
        dst = off + n_curr;

        // 8-wide double: j advances 16
        for (; j + 15 < off + n_curr; j += 16, dst += 8) {
            __m512d d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m512d d_right, e_right, f_right, b_right, dummy;

            deinterleave_pd512(_mm512_loadu_pd(d+j-1), _mm512_loadu_pd(d+j+7),  d_left, d_node);
            deinterleave_pd512(_mm512_loadu_pd(e+j-1), _mm512_loadu_pd(e+j+7),  e_left, e_node);
            deinterleave_pd512(_mm512_loadu_pd(f+j-1), _mm512_loadu_pd(f+j+7),  f_left, f_node);
            deinterleave_pd512(_mm512_loadu_pd(b+j-1), _mm512_loadu_pd(b+j+7),  b_left, b_node);

            deinterleave_pd512(_mm512_loadu_pd(d+j+1), _mm512_loadu_pd(d+j+9),  d_right, dummy);
            deinterleave_pd512(_mm512_loadu_pd(e+j+1), _mm512_loadu_pd(e+j+9),  e_right, dummy);
            deinterleave_pd512(_mm512_loadu_pd(f+j+1), _mm512_loadu_pd(f+j+9),  f_right, dummy);
            deinterleave_pd512(_mm512_loadu_pd(b+j+1), _mm512_loadu_pd(b+j+9),  b_right, dummy);

            __m512d alpha   = _mm512_div_pd(_mm512_sub_pd(_mm512_setzero_pd(), e_node), d_left);
            __m512d gamma_v = _mm512_div_pd(_mm512_sub_pd(_mm512_setzero_pd(), f_node), d_right);

            _mm512_storeu_pd(d+dst, _mm512_fmadd_pd(gamma_v, e_right,
                                    _mm512_fmadd_pd(alpha,   f_left,  d_node)));
            _mm512_storeu_pd(b+dst, _mm512_fmadd_pd(gamma_v, b_right,
                                    _mm512_fmadd_pd(alpha,   b_left,  b_node)));
            _mm512_storeu_pd(e+dst, _mm512_mul_pd(alpha,   e_left));
            _mm512_storeu_pd(f+dst, _mm512_mul_pd(gamma_v, f_right));
        }

        // AVX2 4-wide tail
        for (; j + 7 < off + n_curr; j += 8, dst += 4) {
            __m256d d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m256d d_right, e_right, f_right, b_right, dummy;
            deinterleave_pd(_mm256_loadu_pd(d+j-1), _mm256_loadu_pd(d+j+3), d_left, d_node);
            deinterleave_pd(_mm256_loadu_pd(e+j-1), _mm256_loadu_pd(e+j+3), e_left, e_node);
            deinterleave_pd(_mm256_loadu_pd(f+j-1), _mm256_loadu_pd(f+j+3), f_left, f_node);
            deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3), b_left, b_node);
            deinterleave_pd(_mm256_loadu_pd(d+j+1), _mm256_loadu_pd(d+j+5), d_right, dummy);
            deinterleave_pd(_mm256_loadu_pd(e+j+1), _mm256_loadu_pd(e+j+5), e_right, dummy);
            deinterleave_pd(_mm256_loadu_pd(f+j+1), _mm256_loadu_pd(f+j+5), f_right, dummy);
            deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5), b_right, dummy);
            __m256d zero = _mm256_setzero_pd();
            __m256d alpha   = _mm256_div_pd(_mm256_sub_pd(zero, e_node), d_left);
            __m256d gamma_v = _mm256_div_pd(_mm256_sub_pd(zero, f_node), d_right);
            _mm256_storeu_pd(d+dst, _mm256_fmadd_pd(gamma_v, e_right, _mm256_fmadd_pd(alpha, f_left, d_node)));
            _mm256_storeu_pd(b+dst, _mm256_fmadd_pd(gamma_v, b_right, _mm256_fmadd_pd(alpha, b_left, b_node)));
            _mm256_storeu_pd(e+dst, _mm256_mul_pd(alpha, e_left));
            _mm256_storeu_pd(f+dst, _mm256_mul_pd(gamma_v, f_right));
        }

        for (; j+1 < off+n_curr; j += 2, dst++) {
            double a=-e[j]/d[j-1], g=-f[j]/d[j+1];
            d[dst]=d[j]+a*f[j-1]+g*e[j+1]; b[dst]=b[j]+a*b[j-1]+g*b[j+1];
            e[dst]=a*e[j-1]; f[dst]=g*f[j+1];
        }
        for (; j < off+n_curr; j += 2, dst++) {
            double a=-e[j]/d[j-1];
            d[dst]=d[j]+a*f[j-1]; b[dst]=b[j]+a*b[j-1]; e[dst]=a*e[j-1];
        }
        off += n_curr; n_curr = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr*2+1 : n_curr*2;

        for (j = off, dst = off-n_next+1; j < off+n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j]*b[j+1]) / d[j];
        j += 2;

        for (; j + 15 < off; j += 16) {
            __m512d b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;
            deinterleave_pd512(_mm512_loadu_pd(b+j-1), _mm512_loadu_pd(b+j+7), b_left, b_node);
            deinterleave_pd512(_mm512_loadu_pd(d+j-1), _mm512_loadu_pd(d+j+7), d_left, d_node);
            deinterleave_pd512(_mm512_loadu_pd(e+j-1), _mm512_loadu_pd(e+j+7), e_left, e_node);
            deinterleave_pd512(_mm512_loadu_pd(f+j-1), _mm512_loadu_pd(f+j+7), f_left, f_node);
            __m512d b_right, rdum;
            deinterleave_pd512(_mm512_loadu_pd(b+j+1), _mm512_loadu_pd(b+j+9), b_right, rdum);
            __m512d res = _mm512_div_pd(
                _mm512_fnmadd_pd(f_node, b_right,
                _mm512_fnmadd_pd(e_node, b_left, b_node)), d_node);
            double tmp[8]; _mm512_storeu_pd(tmp, res);
            for (int k = 0; k < 8; k++) b[j + 2*k] = tmp[k];
        }
        for (; j + 7 < off; j += 8) {
            __m256d b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;
            deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3), b_left, b_node);
            deinterleave_pd(_mm256_loadu_pd(d+j-1), _mm256_loadu_pd(d+j+3), d_left, d_node);
            deinterleave_pd(_mm256_loadu_pd(e+j-1), _mm256_loadu_pd(e+j+3), e_left, e_node);
            deinterleave_pd(_mm256_loadu_pd(f+j-1), _mm256_loadu_pd(f+j+3), f_left, f_node);
            __m256d b_right, rdum;
            deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5), b_right, rdum);
            __m256d res = _mm256_div_pd(
                _mm256_fnmadd_pd(f_node, b_right,
                _mm256_fnmadd_pd(e_node, b_left, b_node)), d_node);
            double tmp[4]; _mm256_storeu_pd(tmp, res);
            for (int k = 0; k < 4; k++) b[j + 2*k] = tmp[k];
        }
        for (; j+1 < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1] - f[j]*b[j+1]) / d[j];
        for (; j < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1]) / d[j];

        off -= n_next; n_curr = n_next;
    }
}

// ---- crkgc AVX512 float ----
inline void cyclic_reduction_kershaw_general_continue_avx512(
    float *__restrict d, float *__restrict e, float *__restrict f,
    float *__restrict b, int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;
    off    = 0;
    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;
        j   = off + 1;
        dst = off + n_curr;

        for (; j + 31 < off + n_curr; j += 32, dst += 16) {
            __m512 d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m512 d_right, b_right, dummy;
            deinterleave_ps512(_mm512_loadu_ps(d+j-1), _mm512_loadu_ps(d+j+15), d_left, d_node);
            deinterleave_ps512(_mm512_loadu_ps(e+j-1), _mm512_loadu_ps(e+j+15), e_left, e_node);
            deinterleave_ps512(_mm512_loadu_ps(f+j-1), _mm512_loadu_ps(f+j+15), f_left, f_node);
            deinterleave_ps512(_mm512_loadu_ps(b+j-1), _mm512_loadu_ps(b+j+15), b_left, b_node);
            deinterleave_ps512(_mm512_loadu_ps(d+j+1), _mm512_loadu_ps(d+j+17), d_right, dummy);
            deinterleave_ps512(_mm512_loadu_ps(b+j+1), _mm512_loadu_ps(b+j+17), b_right, dummy);
            __m512 alpha   = _mm512_div_ps(_mm512_sub_ps(_mm512_setzero_ps(), e_node), d_left);
            __m512 gamma_v = _mm512_div_ps(_mm512_sub_ps(_mm512_setzero_ps(), f_node), d_right);
            _mm512_storeu_ps(b+dst, _mm512_fmadd_ps(gamma_v, b_right,
                                    _mm512_fmadd_ps(alpha,   b_left,  b_node)));
        }
        for (; j + 15 < off + n_curr; j += 16, dst += 8) {
            __m256 d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m256 d_right, b_right, dummy;
            deinterleave_ps(_mm256_loadu_ps(d+j-1), _mm256_loadu_ps(d+j+7), d_left, d_node);
            deinterleave_ps(_mm256_loadu_ps(e+j-1), _mm256_loadu_ps(e+j+7), e_left, e_node);
            deinterleave_ps(_mm256_loadu_ps(f+j-1), _mm256_loadu_ps(f+j+7), f_left, f_node);
            deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7), b_left, b_node);
            deinterleave_ps(_mm256_loadu_ps(d+j+1), _mm256_loadu_ps(d+j+9), d_right, dummy);
            deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9), b_right, dummy);
            __m256 zero = _mm256_setzero_ps();
            __m256 alpha   = _mm256_div_ps(_mm256_sub_ps(zero, e_node), d_left);
            __m256 gamma_v = _mm256_div_ps(_mm256_sub_ps(zero, f_node), d_right);
            _mm256_storeu_ps(b+dst, _mm256_fmadd_ps(gamma_v, b_right,
                                    _mm256_fmadd_ps(alpha,   b_left,  b_node)));
        }
        for (; j+1 < off+n_curr; j += 2, dst++) {
            float a=-e[j]/d[j-1], g=-f[j]/d[j+1];
            b[dst] = b[j]+a*b[j-1]+g*b[j+1];
        }
        for (; j < off+n_curr; j += 2, dst++) {
            b[dst] = b[j] + (-e[j]/d[j-1])*b[j-1];
        }
        off += n_curr; n_curr = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr*2+1 : n_curr*2;
        for (j = off, dst = off-n_next+1; j < off+n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j]*b[j+1]) / d[j];
        j += 2;

        for (; j + 31 < off; j += 32) {
            __m512 b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;
            deinterleave_ps512(_mm512_loadu_ps(b+j-1), _mm512_loadu_ps(b+j+15), b_left, b_node);
            deinterleave_ps512(_mm512_loadu_ps(d+j-1), _mm512_loadu_ps(d+j+15), d_left, d_node);
            deinterleave_ps512(_mm512_loadu_ps(e+j-1), _mm512_loadu_ps(e+j+15), e_left, e_node);
            deinterleave_ps512(_mm512_loadu_ps(f+j-1), _mm512_loadu_ps(f+j+15), f_left, f_node);
            __m512 b_right, rdum;
            deinterleave_ps512(_mm512_loadu_ps(b+j+1), _mm512_loadu_ps(b+j+17), b_right, rdum);
            __m512 res = _mm512_div_ps(
                _mm512_fnmadd_ps(f_node, b_right,
                _mm512_fnmadd_ps(e_node, b_left, b_node)), d_node);
            float tmp[16]; _mm512_storeu_ps(tmp, res);
            for (int k = 0; k < 16; k++) b[j + 2*k] = tmp[k];
        }
        for (; j + 15 < off; j += 16) {
            __m256 b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;
            deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7), b_left, b_node);
            deinterleave_ps(_mm256_loadu_ps(d+j-1), _mm256_loadu_ps(d+j+7), d_left, d_node);
            deinterleave_ps(_mm256_loadu_ps(e+j-1), _mm256_loadu_ps(e+j+7), e_left, e_node);
            deinterleave_ps(_mm256_loadu_ps(f+j-1), _mm256_loadu_ps(f+j+7), f_left, f_node);
            __m256 b_right, rdum;
            deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9), b_right, rdum);
            __m256 res = _mm256_div_ps(
                _mm256_fnmadd_ps(f_node, b_right,
                _mm256_fnmadd_ps(e_node, b_left, b_node)), d_node);
            float tmp[8]; _mm256_storeu_ps(tmp, res);
            for (int k = 0; k < 8; k++) b[j + 2*k] = tmp[k];
        }
        for (; j+1 < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1] - f[j]*b[j+1]) / d[j];
        for (; j < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1]) / d[j];

        off -= n_next; n_curr = n_next;
    }
}

// ---- crkgc AVX512 double ----
inline void cyclic_reduction_kershaw_general_continue_avx512(
    double *__restrict d, double *__restrict e, double *__restrict f,
    double *__restrict b, int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;
    off    = 0;
    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;
        j   = off + 1;
        dst = off + n_curr;

        for (; j + 15 < off + n_curr; j += 16, dst += 8) {
            __m512d d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m512d d_right, b_right, dummy;
            deinterleave_pd512(_mm512_loadu_pd(d+j-1), _mm512_loadu_pd(d+j+7), d_left, d_node);
            deinterleave_pd512(_mm512_loadu_pd(e+j-1), _mm512_loadu_pd(e+j+7), e_left, e_node);
            deinterleave_pd512(_mm512_loadu_pd(f+j-1), _mm512_loadu_pd(f+j+7), f_left, f_node);
            deinterleave_pd512(_mm512_loadu_pd(b+j-1), _mm512_loadu_pd(b+j+7), b_left, b_node);
            deinterleave_pd512(_mm512_loadu_pd(d+j+1), _mm512_loadu_pd(d+j+9), d_right, dummy);
            deinterleave_pd512(_mm512_loadu_pd(b+j+1), _mm512_loadu_pd(b+j+9), b_right, dummy);
            __m512d alpha   = _mm512_div_pd(_mm512_sub_pd(_mm512_setzero_pd(), e_node), d_left);
            __m512d gamma_v = _mm512_div_pd(_mm512_sub_pd(_mm512_setzero_pd(), f_node), d_right);
            _mm512_storeu_pd(b+dst, _mm512_fmadd_pd(gamma_v, b_right,
                                    _mm512_fmadd_pd(alpha,   b_left,  b_node)));
        }
        for (; j + 7 < off + n_curr; j += 8, dst += 4) {
            __m256d d_left, d_node, e_left, e_node, f_left, f_node, b_left, b_node;
            __m256d d_right, b_right, dummy;
            deinterleave_pd(_mm256_loadu_pd(d+j-1), _mm256_loadu_pd(d+j+3), d_left, d_node);
            deinterleave_pd(_mm256_loadu_pd(e+j-1), _mm256_loadu_pd(e+j+3), e_left, e_node);
            deinterleave_pd(_mm256_loadu_pd(f+j-1), _mm256_loadu_pd(f+j+3), f_left, f_node);
            deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3), b_left, b_node);
            deinterleave_pd(_mm256_loadu_pd(d+j+1), _mm256_loadu_pd(d+j+5), d_right, dummy);
            deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5), b_right, dummy);
            __m256d zero = _mm256_setzero_pd();
            __m256d alpha   = _mm256_div_pd(_mm256_sub_pd(zero, e_node), d_left);
            __m256d gamma_v = _mm256_div_pd(_mm256_sub_pd(zero, f_node), d_right);
            _mm256_storeu_pd(b+dst, _mm256_fmadd_pd(gamma_v, b_right,
                                    _mm256_fmadd_pd(alpha,   b_left,  b_node)));
        }
        for (; j+1 < off+n_curr; j += 2, dst++) {
            double a=-e[j]/d[j-1], g=-f[j]/d[j+1];
            b[dst] = b[j]+a*b[j-1]+g*b[j+1];
        }
        for (; j < off+n_curr; j += 2, dst++) {
            b[dst] = b[j] + (-e[j]/d[j-1])*b[j-1];
        }
        off += n_curr; n_curr = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr*2+1 : n_curr*2;
        for (j = off, dst = off-n_next+1; j < off+n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j]*b[j+1]) / d[j];
        j += 2;

        for (; j + 15 < off; j += 16) {
            __m512d b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;
            deinterleave_pd512(_mm512_loadu_pd(b+j-1), _mm512_loadu_pd(b+j+7), b_left, b_node);
            deinterleave_pd512(_mm512_loadu_pd(d+j-1), _mm512_loadu_pd(d+j+7), d_left, d_node);
            deinterleave_pd512(_mm512_loadu_pd(e+j-1), _mm512_loadu_pd(e+j+7), e_left, e_node);
            deinterleave_pd512(_mm512_loadu_pd(f+j-1), _mm512_loadu_pd(f+j+7), f_left, f_node);
            __m512d b_right, rdum;
            deinterleave_pd512(_mm512_loadu_pd(b+j+1), _mm512_loadu_pd(b+j+9), b_right, rdum);
            __m512d res = _mm512_div_pd(
                _mm512_fnmadd_pd(f_node, b_right,
                _mm512_fnmadd_pd(e_node, b_left, b_node)), d_node);
            double tmp[8]; _mm512_storeu_pd(tmp, res);
            for (int k = 0; k < 8; k++) b[j + 2*k] = tmp[k];
        }
        for (; j + 7 < off; j += 8) {
            __m256d b_left, b_node, d_left, d_node, e_left, e_node, f_left, f_node, dummy;
            deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3), b_left, b_node);
            deinterleave_pd(_mm256_loadu_pd(d+j-1), _mm256_loadu_pd(d+j+3), d_left, d_node);
            deinterleave_pd(_mm256_loadu_pd(e+j-1), _mm256_loadu_pd(e+j+3), e_left, e_node);
            deinterleave_pd(_mm256_loadu_pd(f+j-1), _mm256_loadu_pd(f+j+3), f_left, f_node);
            __m256d b_right, rdum;
            deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5), b_right, rdum);
            __m256d res = _mm256_div_pd(
                _mm256_fnmadd_pd(f_node, b_right,
                _mm256_fnmadd_pd(e_node, b_left, b_node)), d_node);
            double tmp[4]; _mm256_storeu_pd(tmp, res);
            for (int k = 0; k < 4; k++) b[j + 2*k] = tmp[k];
        }
        for (; j+1 < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1] - f[j]*b[j+1]) / d[j];
        for (; j < off; j += 2)
            b[j] = (b[j] - e[j]*b[j-1]) / d[j];

        off -= n_next; n_curr = n_next;
    }
}

// ---- CyclicReductionAVX512<T> (precomputed, 16-wide float / 8-wide double) ----
template<typename T>
class CyclicReductionAVX512 {
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>,
                  "CyclicReductionAVX512 requires float or double");
public:
    explicit CyclicReductionAVX512(int n)
        : n_(n)
        , q_((int)std::ceil(std::log2((double)(n+1))))
        , inv_d_mid_(T(0))
        , alpha_fwd_(2*n+2, T(0)), gamma_fwd_(2*n+2, T(0))
        , inv_d_bs_(2*n+2, T(0)), e_bs_(2*n+2, T(0)), f_bs_(2*n+2, T(0))
    {}

    // Identical scalar prepare to CyclicReductionAVX2.
    void prepare(const T *d_in, const T *e_in, const T *f_in) {
        const int n = n_, q = q_;
        std::vector<T> dk(2*n+2, T(0)), ek(2*n+2, T(0)), fk(2*n+2, T(0));
        for (int i = 0; i < n; i++) { dk[i]=d_in[i]; ek[i]=e_in[i]; fk[i]=f_in[i]; }

        int j, l, n_curr = n, n_next, off = 0, dst;
        for (l = 1; l < q; l++) {
            n_next = (n_curr - n_curr % 2) / 2;
            j = off+1; dst = off+n_curr;
            for (; j+1 < off+n_curr; j += 2, dst++) {
                T a=-ek[j]/dk[j-1], g=-fk[j]/dk[j+1];
                alpha_fwd_[dst]=a; gamma_fwd_[dst]=g;
                dk[dst]=dk[j]+a*fk[j-1]+g*ek[j+1]; ek[dst]=a*ek[j-1]; fk[dst]=g*fk[j+1];
            }
            for (; j < off+n_curr; j += 2, dst++) {
                T a=-ek[j]/dk[j-1];
                alpha_fwd_[dst]=a; gamma_fwd_[dst]=T(0);
                dk[dst]=dk[j]+a*fk[j-1]; ek[dst]=a*ek[j-1]; fk[dst]=T(0);
            }
            off += n_curr; n_curr = n_next;
        }
        inv_d_mid_ = T(1)/dk[off];
        n_curr = 1;
        for (int mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
            n_next = (n & mask) ? n_curr*2+1 : n_curr*2;
            j = off-n_next;
            inv_d_bs_[j]=T(1)/dk[j]; e_bs_[j]=T(0); f_bs_[j]=fk[j]/dk[j];
            j += 2;
            for (; j+1 < off; j += 2) {
                inv_d_bs_[j]=T(1)/dk[j]; e_bs_[j]=ek[j]/dk[j]; f_bs_[j]=fk[j]/dk[j];
            }
            for (; j < off; j += 2) {
                inv_d_bs_[j]=T(1)/dk[j]; e_bs_[j]=ek[j]/dk[j]; f_bs_[j]=T(0);
            }
            off -= n_next; n_curr = n_next;
        }
    }

    void execute(T *b) {
        const int n = n_, q = q_;
        int j, l, n_curr = n, n_next, off = 0, dst, mask;

        for (l = 1; l < q; l++) {
            n_next = (n_curr - n_curr % 2) / 2;
            j = off+1; dst = off+n_curr;

            if constexpr (std::is_same_v<T, float>) {
                for (; j + 31 < off + n_curr; j += 32, dst += 16) {
                    __m512 alpha_v = _mm512_loadu_ps(alpha_fwd_.data() + dst);
                    __m512 gamma_v = _mm512_loadu_ps(gamma_fwd_.data() + dst);
                    __m512 b_left, b_node, b_right, dummy;
                    deinterleave_ps512(_mm512_loadu_ps(b+j-1), _mm512_loadu_ps(b+j+15), b_left, b_node);
                    deinterleave_ps512(_mm512_loadu_ps(b+j+1), _mm512_loadu_ps(b+j+17), b_right, dummy);
                    _mm512_storeu_ps(b+dst, _mm512_fmadd_ps(gamma_v, b_right,
                                            _mm512_fmadd_ps(alpha_v, b_left, b_node)));
                }
                for (; j + 15 < off + n_curr; j += 16, dst += 8) {
                    __m256 alpha_v = _mm256_loadu_ps(alpha_fwd_.data() + dst);
                    __m256 gamma_v = _mm256_loadu_ps(gamma_fwd_.data() + dst);
                    __m256 b_left, b_node, b_right, dummy;
                    deinterleave_ps(_mm256_loadu_ps(b+j-1), _mm256_loadu_ps(b+j+7), b_left, b_node);
                    deinterleave_ps(_mm256_loadu_ps(b+j+1), _mm256_loadu_ps(b+j+9), b_right, dummy);
                    _mm256_storeu_ps(b+dst, _mm256_fmadd_ps(gamma_v, b_right,
                                            _mm256_fmadd_ps(alpha_v, b_left, b_node)));
                }
            } else {
                for (; j + 15 < off + n_curr; j += 16, dst += 8) {
                    __m512d alpha_v = _mm512_loadu_pd(alpha_fwd_.data() + dst);
                    __m512d gamma_v = _mm512_loadu_pd(gamma_fwd_.data() + dst);
                    __m512d b_left, b_node, b_right, dummy;
                    deinterleave_pd512(_mm512_loadu_pd(b+j-1), _mm512_loadu_pd(b+j+7), b_left, b_node);
                    deinterleave_pd512(_mm512_loadu_pd(b+j+1), _mm512_loadu_pd(b+j+9), b_right, dummy);
                    _mm512_storeu_pd(b+dst, _mm512_fmadd_pd(gamma_v, b_right,
                                            _mm512_fmadd_pd(alpha_v, b_left, b_node)));
                }
                for (; j + 7 < off + n_curr; j += 8, dst += 4) {
                    __m256d alpha_v = _mm256_loadu_pd(alpha_fwd_.data() + dst);
                    __m256d gamma_v = _mm256_loadu_pd(gamma_fwd_.data() + dst);
                    __m256d b_left, b_node, b_right, dummy;
                    deinterleave_pd(_mm256_loadu_pd(b+j-1), _mm256_loadu_pd(b+j+3), b_left, b_node);
                    deinterleave_pd(_mm256_loadu_pd(b+j+1), _mm256_loadu_pd(b+j+5), b_right, dummy);
                    _mm256_storeu_pd(b+dst, _mm256_fmadd_pd(gamma_v, b_right,
                                            _mm256_fmadd_pd(alpha_v, b_left, b_node)));
                }
            }

            for (; j+1 < off+n_curr; j += 2, dst++)
                b[dst] = b[j] + alpha_fwd_[dst]*b[j-1] + gamma_fwd_[dst]*b[j+1];
            for (; j < off+n_curr; j += 2, dst++)
                b[dst] = b[j] + alpha_fwd_[dst]*b[j-1];

            off += n_curr; n_curr = n_next;
        }

        b[off] = b[off] * inv_d_mid_;
        n_curr = 1;

        for (mask = (1<<(q-1))>>1; mask > 0; mask >>= 1) {
            n_next = (n & mask) ? n_curr*2+1 : n_curr*2;
            for (j = off, dst = off-n_next+1; j < off+n_curr; j++, dst += 2)
                b[dst] = b[j];

            j = off - n_next;
            b[j] = inv_d_bs_[j]*b[j] - f_bs_[j]*b[j+1];
            j += 2;

            if constexpr (std::is_same_v<T, float>) {
                for (; j + 31 < off; j += 32) {
                    __m512 b_left, b_node, b_right, dummy;
                    __m512 inv_d_v, e_v, f_v, junk;
                    deinterleave_ps512(_mm512_loadu_ps(b+j-1),           _mm512_loadu_ps(b+j+15),           b_left, b_node);
                    deinterleave_ps512(_mm512_loadu_ps(b+j+1),           _mm512_loadu_ps(b+j+17),           b_right, dummy);
                    deinterleave_ps512(_mm512_loadu_ps(inv_d_bs_.data()+j-1), _mm512_loadu_ps(inv_d_bs_.data()+j+15), junk, inv_d_v);
                    deinterleave_ps512(_mm512_loadu_ps(e_bs_.data()+j-1),     _mm512_loadu_ps(e_bs_.data()+j+15),     junk, e_v);
                    deinterleave_ps512(_mm512_loadu_ps(f_bs_.data()+j-1),     _mm512_loadu_ps(f_bs_.data()+j+15),     junk, f_v);
                    __m512 res = _mm512_fnmadd_ps(f_v, b_right,
                                 _mm512_fnmadd_ps(e_v, b_left,
                                 _mm512_mul_ps(inv_d_v, b_node)));
                    float tmp[16]; _mm512_storeu_ps(tmp, res);
                    for (int k = 0; k < 16; k++) b[j + 2*k] = tmp[k];
                }
                for (; j + 15 < off; j += 16) {
                    __m256 b_left, b_node, b_right, dummy;
                    __m256 inv_d_v, e_v, f_v, junk;
                    deinterleave_ps(_mm256_loadu_ps(b+j-1),           _mm256_loadu_ps(b+j+7),           b_left, b_node);
                    deinterleave_ps(_mm256_loadu_ps(b+j+1),           _mm256_loadu_ps(b+j+9),           b_right, dummy);
                    deinterleave_ps(_mm256_loadu_ps(inv_d_bs_.data()+j-1), _mm256_loadu_ps(inv_d_bs_.data()+j+7), junk, inv_d_v);
                    deinterleave_ps(_mm256_loadu_ps(e_bs_.data()+j-1),     _mm256_loadu_ps(e_bs_.data()+j+7),     junk, e_v);
                    deinterleave_ps(_mm256_loadu_ps(f_bs_.data()+j-1),     _mm256_loadu_ps(f_bs_.data()+j+7),     junk, f_v);
                    __m256 res = _mm256_fnmadd_ps(f_v, b_right,
                                 _mm256_fnmadd_ps(e_v, b_left,
                                 _mm256_mul_ps(inv_d_v, b_node)));
                    float tmp[8]; _mm256_storeu_ps(tmp, res);
                    for (int k = 0; k < 8; k++) b[j + 2*k] = tmp[k];
                }
            } else {
                for (; j + 15 < off; j += 16) {
                    __m512d b_left, b_node, b_right, dummy;
                    __m512d inv_d_v, e_v, f_v, junk;
                    deinterleave_pd512(_mm512_loadu_pd(b+j-1),           _mm512_loadu_pd(b+j+7),           b_left, b_node);
                    deinterleave_pd512(_mm512_loadu_pd(b+j+1),           _mm512_loadu_pd(b+j+9),           b_right, dummy);
                    deinterleave_pd512(_mm512_loadu_pd(inv_d_bs_.data()+j-1), _mm512_loadu_pd(inv_d_bs_.data()+j+7), junk, inv_d_v);
                    deinterleave_pd512(_mm512_loadu_pd(e_bs_.data()+j-1),     _mm512_loadu_pd(e_bs_.data()+j+7),     junk, e_v);
                    deinterleave_pd512(_mm512_loadu_pd(f_bs_.data()+j-1),     _mm512_loadu_pd(f_bs_.data()+j+7),     junk, f_v);
                    __m512d res = _mm512_fnmadd_pd(f_v, b_right,
                                  _mm512_fnmadd_pd(e_v, b_left,
                                  _mm512_mul_pd(inv_d_v, b_node)));
                    double tmp[8]; _mm512_storeu_pd(tmp, res);
                    for (int k = 0; k < 8; k++) b[j + 2*k] = tmp[k];
                }
                for (; j + 7 < off; j += 8) {
                    __m256d b_left, b_node, b_right, dummy;
                    __m256d inv_d_v, e_v, f_v, junk;
                    deinterleave_pd(_mm256_loadu_pd(b+j-1),           _mm256_loadu_pd(b+j+3),           b_left, b_node);
                    deinterleave_pd(_mm256_loadu_pd(b+j+1),           _mm256_loadu_pd(b+j+5),           b_right, dummy);
                    deinterleave_pd(_mm256_loadu_pd(inv_d_bs_.data()+j-1), _mm256_loadu_pd(inv_d_bs_.data()+j+3), junk, inv_d_v);
                    deinterleave_pd(_mm256_loadu_pd(e_bs_.data()+j-1),     _mm256_loadu_pd(e_bs_.data()+j+3),     junk, e_v);
                    deinterleave_pd(_mm256_loadu_pd(f_bs_.data()+j-1),     _mm256_loadu_pd(f_bs_.data()+j+3),     junk, f_v);
                    __m256d res = _mm256_fnmadd_pd(f_v, b_right,
                                  _mm256_fnmadd_pd(e_v, b_left,
                                  _mm256_mul_pd(inv_d_v, b_node)));
                    double tmp[4]; _mm256_storeu_pd(tmp, res);
                    for (int k = 0; k < 4; k++) b[j + 2*k] = tmp[k];
                }
            }

            for (; j+1 < off; j += 2)
                b[j] = inv_d_bs_[j]*b[j] - e_bs_[j]*b[j-1] - f_bs_[j]*b[j+1];
            for (; j < off; j += 2)
                b[j] = inv_d_bs_[j]*b[j] - e_bs_[j]*b[j-1];

            off -= n_next; n_curr = n_next;
        }
    }

private:
    int n_, q_;
    T   inv_d_mid_;
    std::vector<T> alpha_fwd_, gamma_fwd_;
    std::vector<T> inv_d_bs_, e_bs_, f_bs_;
};

#endif // __AVX512F__

} // namespace fdm
#endif // __AVX2__
