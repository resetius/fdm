#pragma once

#ifdef __ARM_NEON
#include <arm_neon.h>

namespace fdm {

// NEON-vectorized cyclic reduction (Kershaw general layout) for float.
//
// Forward sweep: processes 4 iterations at once with float32x4 (vld2q_f32
// deinterleaves the alternating even/odd layout).  Stores are contiguous
// (destination block is separate from the source block).
//
// Back-substitution: even positions are independent once odd positions are
// filled by the copy step, so we vectorize those too (scalar scatter for
// the 4 results – compute is still 4-wide).
//
// Requires AArch64 (M1/M2/…) for vdivq_f32 / vdivq_f64.
inline void cyclic_reduction_kershaw_general_neon(
    float *__restrict d, float *__restrict e, float *__restrict f,
    float *__restrict b,
    int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;

    off    = 0;
    n_curr = n;

    // ---- forward sweep ----
    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;

        j   = off + 1;
        dst = off + n_curr;

        // 4-wide SIMD block: handle j, j+2, j+4, j+6 simultaneously.
        // Condition: all four must satisfy the original (x+1 < off+n_curr),
        // so the last one j+6 requires j+7 < off+n_curr.
        // vld2q_f32(ptr+j-1) deinterleaves 8 floats:
        //   val[0] = { ptr[j-1], ptr[j+1], ptr[j+3], ptr[j+5] }  (even offsets = neighbours)
        //   val[1] = { ptr[j],   ptr[j+2], ptr[j+4], ptr[j+6] }  (odd  offsets = nodes)
        // vld2q_f32(ptr+j+1).val[0] = { ptr[j+1], ptr[j+3], ptr[j+5], ptr[j+7] }
        //                              = right-neighbour of each node.
        for (; j + 7 < off + n_curr; j += 8, dst += 4) {
            float32x4x2_t dv = vld2q_f32(d + j - 1);
            float32x4x2_t ev = vld2q_f32(e + j - 1);
            float32x4x2_t fv = vld2q_f32(f + j - 1);
            float32x4x2_t bv = vld2q_f32(b + j - 1);

            float32x4_t d_jp1 = vld2q_f32(d + j + 1).val[0]; // d[j+1]
            float32x4_t e_jp1 = vld2q_f32(e + j + 1).val[0]; // e[j+1]
            float32x4_t f_jp1 = vld2q_f32(f + j + 1).val[0]; // f[j+1]
            float32x4_t b_jp1 = vld2q_f32(b + j + 1).val[0]; // b[j+1]

            // alpha = -e[j] / d[j-1],  gamma = -f[j] / d[j+1]
            float32x4_t alpha   = vdivq_f32(vnegq_f32(ev.val[1]), dv.val[0]);
            float32x4_t gamma_v = vdivq_f32(vnegq_f32(fv.val[1]), d_jp1);

            // d[dst] = d[j] + alpha*f[j-1] + gamma*e[j+1]
            float32x4_t d_out = vmlaq_f32(vmlaq_f32(dv.val[1], alpha, fv.val[0]), gamma_v, e_jp1);
            // b[dst] = b[j] + alpha*b[j-1] + gamma*b[j+1]
            float32x4_t b_out = vmlaq_f32(vmlaq_f32(bv.val[1], alpha, bv.val[0]), gamma_v, b_jp1);
            // e[dst] = alpha * e[j-1]
            float32x4_t e_out = vmulq_f32(alpha, ev.val[0]);
            // f[dst] = gamma * f[j+1]
            float32x4_t f_out = vmulq_f32(gamma_v, f_jp1);

            // Destination is contiguous – simple sequential stores.
            vst1q_f32(d + dst, d_out);
            vst1q_f32(b + dst, b_out);
            vst1q_f32(e + dst, e_out);
            vst1q_f32(f + dst, f_out);
        }

        // Scalar tail (fewer than 4 remaining iterations).
        for (; j + 1 < off + n_curr; j += 2, dst++) {
            float alpha = -e[j] / d[j - 1];
            float gamma = -f[j] / d[j + 1];
            d[dst] = d[j] + alpha * f[j - 1] + gamma * e[j + 1];
            b[dst] = b[j] + alpha * b[j - 1] + gamma * b[j + 1];
            e[dst] = alpha * e[j - 1];
            f[dst] = gamma * f[j + 1];
        }

        // Boundary element when n is not a power of 2.
        for (; j < off + n_curr; j += 2, dst++) {
            float alpha = -e[j] / d[j - 1];
            d[dst] = d[j] + alpha * f[j - 1];
            b[dst] = b[j] + alpha * b[j - 1];
            e[dst] = alpha * e[j - 1];
        }

        off    += n_curr;
        n_curr  = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    // ---- back-substitution ----
    for (mask = (1 << (q - 1)) >> 1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr * 2 + 1 : n_curr * 2;

        // Copy known values from current level to odd positions of next level.
        for (j = off, dst = off - n_next + 1; j < off + n_curr; j++, dst += 2)
            b[dst] = b[j];

        // First element: only a right neighbour.
        j = off - n_next;
        b[j] = (b[j] - f[j] * b[j + 1]) / d[j];
        j += 2;

        // Remaining even positions: b[j] depends only on b[j-1] and b[j+1],
        // which are odd (already filled above) ⟹ all 4 updates are independent.
        // 4-wide SIMD: load interleaved pairs, compute, scatter-store.
        for (; j + 7 < off; j += 8) {
            // val[0] = odd  = { b[j-1], b[j+1], b[j+3], b[j+5] }  (known)
            // val[1] = even = { b[j],   b[j+2], b[j+4], b[j+6] }  (forward-sweep values)
            float32x4x2_t bv   = vld2q_f32(b + j - 1);
            float32x4_t b_jp1v = vld2q_f32(b + j + 1).val[0]; // { b[j+1], b[j+3], b[j+5], b[j+7] }
            float32x4x2_t ev   = vld2q_f32(e + j - 1);
            float32x4x2_t fv   = vld2q_f32(f + j - 1);
            float32x4x2_t dv   = vld2q_f32(d + j - 1);

            // num = b[j] - e[j]*b[j-1] - f[j]*b[j+1]
            float32x4_t num = bv.val[1];
            num = vsubq_f32(num, vmulq_f32(ev.val[1], bv.val[0]));
            num = vsubq_f32(num, vmulq_f32(fv.val[1], b_jp1v));
            float32x4_t res = vdivq_f32(num, dv.val[1]);

            // Scatter to even positions (stride 2).
            b[j]   = vgetq_lane_f32(res, 0);
            b[j+2] = vgetq_lane_f32(res, 1);
            b[j+4] = vgetq_lane_f32(res, 2);
            b[j+6] = vgetq_lane_f32(res, 3);
        }

        for (; j + 1 < off; j += 2)
            b[j] = (b[j] - e[j] * b[j - 1] - f[j] * b[j + 1]) / d[j];

        for (; j < off; j += 2)
            b[j] = (b[j] - e[j] * b[j - 1]) / d[j];

        off    -= n_next;
        n_curr  = n_next;
    }
}

// NEON-vectorized cyclic reduction (Kershaw general layout) for double.
// Same structure; float64x2 gives 2-wide parallelism.
inline void cyclic_reduction_kershaw_general_neon(
    double *__restrict d, double *__restrict e, double *__restrict f,
    double *__restrict b,
    int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;

    off    = 0;
    n_curr = n;

    // ---- forward sweep ----
    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;

        j   = off + 1;
        dst = off + n_curr;

        // 2-wide SIMD: handle j and j+2 simultaneously.
        // Condition: j+2 must satisfy j+2+1 < off+n_curr  →  j+3 < off+n_curr.
        // vld2q_f64(ptr+j-1) deinterleaves 4 doubles:
        //   val[0] = { ptr[j-1], ptr[j+1] }  (left neighbours for j, j+2)
        //   val[1] = { ptr[j],   ptr[j+2] }  (nodes)
        for (; j + 3 < off + n_curr; j += 4, dst += 2) {
            float64x2x2_t dv = vld2q_f64(d + j - 1);
            float64x2x2_t ev = vld2q_f64(e + j - 1);
            float64x2x2_t fv = vld2q_f64(f + j - 1);
            float64x2x2_t bv = vld2q_f64(b + j - 1);

            float64x2_t d_jp1 = vld2q_f64(d + j + 1).val[0];
            float64x2_t e_jp1 = vld2q_f64(e + j + 1).val[0];
            float64x2_t f_jp1 = vld2q_f64(f + j + 1).val[0];
            float64x2_t b_jp1 = vld2q_f64(b + j + 1).val[0];

            float64x2_t alpha   = vdivq_f64(vnegq_f64(ev.val[1]), dv.val[0]);
            float64x2_t gamma_v = vdivq_f64(vnegq_f64(fv.val[1]), d_jp1);

            float64x2_t d_out = vmlaq_f64(vmlaq_f64(dv.val[1], alpha, fv.val[0]), gamma_v, e_jp1);
            float64x2_t b_out = vmlaq_f64(vmlaq_f64(bv.val[1], alpha, bv.val[0]), gamma_v, b_jp1);
            float64x2_t e_out = vmulq_f64(alpha, ev.val[0]);
            float64x2_t f_out = vmulq_f64(gamma_v, f_jp1);

            vst1q_f64(d + dst, d_out);
            vst1q_f64(b + dst, b_out);
            vst1q_f64(e + dst, e_out);
            vst1q_f64(f + dst, f_out);
        }

        for (; j + 1 < off + n_curr; j += 2, dst++) {
            double alpha = -e[j] / d[j - 1];
            double gamma = -f[j] / d[j + 1];
            d[dst] = d[j] + alpha * f[j - 1] + gamma * e[j + 1];
            b[dst] = b[j] + alpha * b[j - 1] + gamma * b[j + 1];
            e[dst] = alpha * e[j - 1];
            f[dst] = gamma * f[j + 1];
        }

        for (; j < off + n_curr; j += 2, dst++) {
            double alpha = -e[j] / d[j - 1];
            d[dst] = d[j] + alpha * f[j - 1];
            b[dst] = b[j] + alpha * b[j - 1];
            e[dst] = alpha * e[j - 1];
        }

        off    += n_curr;
        n_curr  = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    // ---- back-substitution ----
    for (mask = (1 << (q - 1)) >> 1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr * 2 + 1 : n_curr * 2;

        for (j = off, dst = off - n_next + 1; j < off + n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j] * b[j + 1]) / d[j];
        j += 2;

        // 2-wide SIMD for independent even positions.
        for (; j + 3 < off; j += 4) {
            float64x2x2_t bv   = vld2q_f64(b + j - 1);
            float64x2_t b_jp1v = vld2q_f64(b + j + 1).val[0];
            float64x2x2_t ev   = vld2q_f64(e + j - 1);
            float64x2x2_t fv   = vld2q_f64(f + j - 1);
            float64x2x2_t dv   = vld2q_f64(d + j - 1);

            float64x2_t num = bv.val[1];
            num = vsubq_f64(num, vmulq_f64(ev.val[1], bv.val[0]));
            num = vsubq_f64(num, vmulq_f64(fv.val[1], b_jp1v));
            float64x2_t res = vdivq_f64(num, dv.val[1]);

            b[j]   = vgetq_lane_f64(res, 0);
            b[j+2] = vgetq_lane_f64(res, 1);
        }

        for (; j + 1 < off; j += 2)
            b[j] = (b[j] - e[j] * b[j - 1] - f[j] * b[j + 1]) / d[j];

        for (; j < off; j += 2)
            b[j] = (b[j] - e[j] * b[j - 1]) / d[j];

        off    -= n_next;
        n_curr  = n_next;
    }
}

// NEON-vectorized _continue variant (Kershaw general layout) for float.
//
// Assumes d/e/f have already been factorized by a prior call to
// cyclic_reduction_kershaw_general (the "precompute" step).
// Only b is updated; d/e/f are read but never written.
// Compared to the full crkg_neon:  saves 3 vst1q + 2 vld2q per SIMD block
// in the forward sweep (no d/e/f writes, no e_jp1/f_jp1 loads).
inline void cyclic_reduction_kershaw_general_continue_neon(
    float *__restrict d, float *__restrict e, float *__restrict f,
    float *__restrict b,
    int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;

    off    = 0;
    n_curr = n;

    // ---- forward sweep (b only) ----
    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;

        j   = off + 1;
        dst = off + n_curr;

        // 4-wide: j, j+2, j+4, j+6
        // Loads needed: d[j-1], d[j+1] (for alpha/gamma denominators),
        //               e[j], f[j] (numerators), b[j-1], b[j], b[j+1].
        // No e_jp1 / f_jp1 loads, no d/e/f writes.
        for (; j + 7 < off + n_curr; j += 8, dst += 4) {
            float32x4x2_t dv  = vld2q_f32(d + j - 1); // val[0]=d[j-1], val[1]=d[j]
            float32x4x2_t ev  = vld2q_f32(e + j - 1); // val[1]=e[j]
            float32x4x2_t fv  = vld2q_f32(f + j - 1); // val[1]=f[j]
            float32x4x2_t bv  = vld2q_f32(b + j - 1); // val[0]=b[j-1], val[1]=b[j]
            float32x4_t d_jp1 = vld2q_f32(d + j + 1).val[0]; // d[j+1]
            float32x4_t b_jp1 = vld2q_f32(b + j + 1).val[0]; // b[j+1]

            float32x4_t alpha   = vdivq_f32(vnegq_f32(ev.val[1]), dv.val[0]);
            float32x4_t gamma_v = vdivq_f32(vnegq_f32(fv.val[1]), d_jp1);

            // b[dst] = b[j] + alpha*b[j-1] + gamma*b[j+1]
            float32x4_t b_out = vmlaq_f32(vmlaq_f32(bv.val[1], alpha, bv.val[0]), gamma_v, b_jp1);
            vst1q_f32(b + dst, b_out); // only b written
        }

        for (; j + 1 < off + n_curr; j += 2, dst++) {
            float alpha = -e[j] / d[j - 1];
            float gamma = -f[j] / d[j + 1];
            b[dst] = b[j] + alpha * b[j - 1] + gamma * b[j + 1];
        }
        for (; j < off + n_curr; j += 2, dst++) {
            float alpha = -e[j] / d[j - 1];
            b[dst] = b[j] + alpha * b[j - 1];
        }

        off    += n_curr;
        n_curr  = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    // ---- back-substitution (identical to full NEON version) ----
    for (mask = (1 << (q - 1)) >> 1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr * 2 + 1 : n_curr * 2;

        for (j = off, dst = off - n_next + 1; j < off + n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j] * b[j + 1]) / d[j];
        j += 2;

        for (; j + 7 < off; j += 8) {
            float32x4x2_t bv   = vld2q_f32(b + j - 1);
            float32x4_t b_jp1v = vld2q_f32(b + j + 1).val[0];
            float32x4x2_t ev   = vld2q_f32(e + j - 1);
            float32x4x2_t fv   = vld2q_f32(f + j - 1);
            float32x4x2_t dv   = vld2q_f32(d + j - 1);

            float32x4_t num = bv.val[1];
            num = vsubq_f32(num, vmulq_f32(ev.val[1], bv.val[0]));
            num = vsubq_f32(num, vmulq_f32(fv.val[1], b_jp1v));
            float32x4_t res = vdivq_f32(num, dv.val[1]);

            b[j]   = vgetq_lane_f32(res, 0);
            b[j+2] = vgetq_lane_f32(res, 1);
            b[j+4] = vgetq_lane_f32(res, 2);
            b[j+6] = vgetq_lane_f32(res, 3);
        }

        for (; j + 1 < off; j += 2)
            b[j] = (b[j] - e[j] * b[j - 1] - f[j] * b[j + 1]) / d[j];
        for (; j < off; j += 2)
            b[j] = (b[j] - e[j] * b[j - 1]) / d[j];

        off    -= n_next;
        n_curr  = n_next;
    }
}

// double version
inline void cyclic_reduction_kershaw_general_continue_neon(
    double *__restrict d, double *__restrict e, double *__restrict f,
    double *__restrict b,
    int q, int n)
{
    int j, l, n_curr, n_next, off, dst, mask;

    off    = 0;
    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;

        j   = off + 1;
        dst = off + n_curr;

        for (; j + 3 < off + n_curr; j += 4, dst += 2) {
            float64x2x2_t dv  = vld2q_f64(d + j - 1);
            float64x2x2_t ev  = vld2q_f64(e + j - 1);
            float64x2x2_t fv  = vld2q_f64(f + j - 1);
            float64x2x2_t bv  = vld2q_f64(b + j - 1);
            float64x2_t d_jp1 = vld2q_f64(d + j + 1).val[0];
            float64x2_t b_jp1 = vld2q_f64(b + j + 1).val[0];

            float64x2_t alpha   = vdivq_f64(vnegq_f64(ev.val[1]), dv.val[0]);
            float64x2_t gamma_v = vdivq_f64(vnegq_f64(fv.val[1]), d_jp1);

            float64x2_t b_out = vmlaq_f64(vmlaq_f64(bv.val[1], alpha, bv.val[0]), gamma_v, b_jp1);
            vst1q_f64(b + dst, b_out);
        }

        for (; j + 1 < off + n_curr; j += 2, dst++) {
            double alpha = -e[j] / d[j - 1];
            double gamma = -f[j] / d[j + 1];
            b[dst] = b[j] + alpha * b[j - 1] + gamma * b[j + 1];
        }
        for (; j < off + n_curr; j += 2, dst++) {
            double alpha = -e[j] / d[j - 1];
            b[dst] = b[j] + alpha * b[j - 1];
        }

        off    += n_curr;
        n_curr  = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1 << (q - 1)) >> 1; mask > 0; mask >>= 1) {
        n_next = (n & mask) ? n_curr * 2 + 1 : n_curr * 2;

        for (j = off, dst = off - n_next + 1; j < off + n_curr; j++, dst += 2)
            b[dst] = b[j];

        j = off - n_next;
        b[j] = (b[j] - f[j] * b[j + 1]) / d[j];
        j += 2;

        for (; j + 3 < off; j += 4) {
            float64x2x2_t bv   = vld2q_f64(b + j - 1);
            float64x2_t b_jp1v = vld2q_f64(b + j + 1).val[0];
            float64x2x2_t ev   = vld2q_f64(e + j - 1);
            float64x2x2_t fv   = vld2q_f64(f + j - 1);
            float64x2x2_t dv   = vld2q_f64(d + j - 1);

            float64x2_t num = bv.val[1];
            num = vsubq_f64(num, vmulq_f64(ev.val[1], bv.val[0]));
            num = vsubq_f64(num, vmulq_f64(fv.val[1], b_jp1v));
            float64x2_t res = vdivq_f64(num, dv.val[1]);

            b[j]   = vgetq_lane_f64(res, 0);
            b[j+2] = vgetq_lane_f64(res, 1);
        }

        for (; j + 1 < off; j += 2)
            b[j] = (b[j] - e[j] * b[j - 1] - f[j] * b[j + 1]) / d[j];
        for (; j < off; j += 2)
            b[j] = (b[j] - e[j] * b[j - 1]) / d[j];

        off    -= n_next;
        n_curr  = n_next;
    }
}

} // namespace fdm
#endif // __ARM_NEON
