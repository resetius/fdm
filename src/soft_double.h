#pragma once

/*
BSD 3-Clause License

Copyright (c) 2026, Alexey Ozeritskiy
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
https://github.com/resetius/fdm/blob/master/src/soft_double.h
*/

#include <cstdint>
#include <stdint.h>
#include <bit>
#include <tuple>

class SoftDouble {
public:
    SoftDouble() : Value(0) {}

    explicit SoftDouble(double d)
        : Value(std::bit_cast<uint64_t>(d))
    { }

    explicit SoftDouble(uint64_t value)
        : Value(value)
    { }

    operator double() const {
        return std::bit_cast<double>(Value);
    }

    explicit SoftDouble(float f)
    {
        uint32_t bits = std::bit_cast<uint32_t>(f);

        uint64_t sign = (uint64_t)(bits >> 31) & 1u;
        uint32_t exp_f = (bits >> 23) & 0xFFu;
        uint32_t frac_f = bits & 0x007FFFFFu;

        // Zero (preserve sign: +0 / -0)
        if (exp_f == 0 && frac_f == 0) {
            Value = sign << 63;
            return;
        }

        // Inf / NaN
        if (exp_f == 0xFFu) {
            uint64_t exp_d = 0x7FFull;
            uint64_t frac_d = 0;

            if (frac_f != 0) {
                // Propagate payload, make it quiet NaN (set top mantissa bit in double)
                frac_d = (uint64_t)frac_f << (52 - 23);
                frac_d |= (1ull << 51); // qNaN bit
            }
            Value = (sign << 63) | (exp_d << 52) | (frac_d & ((1ull << 52) - 1));
            return;
        }

        // Normal float
        if (exp_f != 0) {
            uint64_t exp_d = (uint64_t)exp_f + (1023 - 127);
            uint64_t frac_d = (uint64_t)frac_f << (52 - 23); // just widen fraction
            Value = (sign << 63) | (exp_d << 52) | (frac_d & ((1ull << 52) - 1));
            return;
        }

        // Subnormal float: exp_f == 0, frac_f != 0
        // Value = frac_f * 2^-149. Convert to normalized double.
        // Normalize frac_f so that highest 1 ends up at bit 23 (implicit 1 position).
        uint32_t m = frac_f;
        int shift = 0;
        while ((m & 0x00800000u) == 0) { // while leading 1 not in bit 23
            m <<= 1;
            ++shift;
        }
        // Now m has implicit leading 1 at bit 23. Remove it for fraction field.
        m &= 0x007FFFFFu;

        // For float subnormal: exponent is effectively (1 - 127) - shift, with mantissa in [1,2)
        int exp2 = (1 - 127) - shift;          // unbiased exponent in base2 for the normalized value
        uint64_t exp_d = (uint64_t)(exp2 + 1023);

        uint64_t frac_d = (uint64_t)m << (52 - 23);
        Value = (sign << 63) | (exp_d << 52) | (frac_d & ((1ull << 52) - 1));
    }

    operator float() const {
        const uint64_t sign = (Value >> 63) & 1ull;
        const uint64_t exp_d = (Value >> 52) & 0x7FFull;
        const uint64_t frac_d = Value & ((1ull << 52) - 1);

        auto pack = [&](uint32_t exp_f, uint32_t frac_f) -> float {
            uint32_t bits = (uint32_t)(sign << 31) | (exp_f << 23) | (frac_f & 0x7FFFFFu);
            return std::bit_cast<float>(bits);
        };

        // Zero / subnormal double
        if (exp_d == 0) {
            if (frac_d == 0) {
                // preserve sign of zero
                return pack(0, 0);
            }
            // double subnormal: value = frac_d * 2^-1074
            // Convert to float: may become subnormal or zero. We do rounding.
            // We need significand as integer (no implicit 1 here).
            uint64_t sig = frac_d; // 52-bit
            // We want to shift sig to fit into float subnormal mantissa (23 bits), with exponent fixed to 0.
            // Effective exponent for sig is -1074, and sig has its top bit somewhere <= 51.
            // Normalize sig so top bit is at 52-ish position for easier handling:
            int lz = 0;
            {
                uint64_t t = sig;
                while ((t & (1ull << 51)) == 0) { t <<= 1; ++lz; if (lz > 51) break; }
            }
            // After shifting left by lz, the top bit sits at bit 51. Exponent increases accordingly.
            // Now the value is: (1.xxx) * 2^(-1074 - lz + 51) = (1.xxx) * 2^(-1023 - lz)
            int e2 = -1023 - lz; // unbiased exponent after normalization (like having implicit 1)

            // For float subnormal, unbiased exponent is fixed at -126, and mantissa encodes value:
            // value = (mantissa / 2^23) * 2^-126  => mantissa = value * 2^(149)
            // So we need mantissa = round(value * 2^149).
            // With normalized sig (top at 51), value = (sig_norm / 2^51) * 2^e2
            // => value * 2^149 = sig_norm * 2^(e2 + 149 - 51) = sig_norm * 2^(e2 + 98)
            uint64_t sig_norm = frac_d << lz; // top at bit 51
            int shift = -(e2 + 98); // if positive -> right shift
            if (shift >= 64) {
                return pack(0, 0); // underflow to zero
            }

            uint64_t mant;
            if (shift > 0) {
                // round to nearest even on right shift
                uint64_t dropped_mask = (shift >= 64) ? ~0ull : ((1ull << shift) - 1);
                uint64_t dropped = sig_norm & dropped_mask;
                mant = sig_norm >> shift;

                uint64_t half = 1ull << (shift - 1);
                bool round_up = (dropped > half) || (dropped == half && (mant & 1ull));
                if (round_up) mant += 1;
            } else {
                // left shift
                int l = -shift;
                if (l >= 64) return pack(0, 0);
                mant = sig_norm << l;
            }

            // mant is in "units of 1" where we want 23-bit fraction (no implicit 1 for subnormal)
            // clamp to 23 bits: if it overflows, it becomes the smallest normal float (exp=1, frac=0)
            if (mant >= (1ull << 23)) {
                return pack(1u, 0u);
            }
            return pack(0u, (uint32_t)mant);
        }

        // Inf / NaN
        if (exp_d == 0x7FFull) {
            if (frac_d == 0) {
                return pack(0xFFu, 0u); // inf
            }
            // NaN: propagate payload into float mantissa, make it quiet
            uint32_t payload = (uint32_t)(frac_d >> (52 - 23));
            if ((payload & 0x7FFFFFu) == 0) payload = 1; // ensure mantissa non-zero
            payload |= 0x400000u; // quiet NaN in float (MSB of mantissa)
            return pack(0xFFu, payload);
        }

        // Normal double
        // value = (1.frac_d) * 2^(exp_d-1023)
        int e2 = (int)exp_d - 1023; // unbiased exponent

        // Fast overflow to inf: float max exponent is +127, so if e2 > 127 => inf
        if (e2 > 127) {
            return pack(0xFFu, 0u);
        }

        // Build 53-bit significand with implicit 1
        uint64_t sig = (1ull << 52) | frac_d; // 53 bits

        // Target float:
        // - normal if e2 >= -126
        // - subnormal if e2 < -126 (but maybe still non-zero)
        if (e2 >= -126) {
            // float normal: exp_f = e2 + 127
            // need 24 bits (1 + 23) from sig with rounding from 53 -> 24
            // Take top 24 bits of sig (aligned so that implicit 1 ends up at bit 23)
            // sig has implicit 1 at bit 52, so shift right by 52-23 = 29 to get 24 bits.
            const int shift = 52 - 23; // 29
            uint64_t mant24 = sig >> shift; // 24 bits (includes leading 1)
            uint64_t rem = sig & ((1ull << shift) - 1);

            // round-to-nearest-even
            uint64_t half = 1ull << (shift - 1);
            bool round_up = (rem > half) || (rem == half && (mant24 & 1ull));
            if (round_up) mant24 += 1;

            // handle carry (e.g., 1.111.. rounds to 10.000..)
            if (mant24 == (1ull << 24)) {
                mant24 >>= 1;
                e2 += 1;
                if (e2 > 127) return pack(0xFFu, 0u);
            }

            uint32_t exp_f = (uint32_t)(e2 + 127);
            uint32_t frac_f = (uint32_t)(mant24 & 0x7FFFFFu); // drop implicit 1
            return pack(exp_f, frac_f);
        } else {
            // float subnormal: exp_f = 0, value = mant * 2^-149
            // Need mant = round( (1.frac) * 2^(e2+149) )
            // sig represents (1.frac) * 2^52, so:
            // mant = round( sig * 2^(e2+149-52) ) = round( sig * 2^(e2+97) )
            int sh = -(e2 + 97); // right shift amount if positive
            if (sh >= 64) {
                return pack(0u, 0u); // too small -> 0
            }

            uint64_t mant;
            if (sh > 0) {
                uint64_t dropped_mask = (sh >= 64) ? ~0ull : ((1ull << sh) - 1);
                uint64_t dropped = sig & dropped_mask;
                mant = sig >> sh;

                uint64_t half = 1ull << (sh - 1);
                bool round_up = (dropped > half) || (dropped == half && (mant & 1ull));
                if (round_up) mant += 1;
            } else {
                int l = -sh;
                if (l >= 64) return pack(0u, 0u);
                mant = sig << l;
            }

            // If rounding produced 2^23, it becomes the smallest normal float
            if (mant >= (1ull << 23)) {
                return pack(1u, 0u);
            }
            return pack(0u, (uint32_t)mant);
        }
    }

    SoftDouble& operator=(const SoftDouble& other) {
        Value = other.Value;
        return *this;
    }

    SoftDouble operator+(const SoftDouble& other) const {
        return SoftDouble{AddUnchecked(Value, other.Value)};
    }

private:
    static uint64_t AddUnchecked(uint64_t left, uint64_t right) {
        static constexpr uint64_t hidden = 1ULL << 52;
        static constexpr uint64_t carry = 1ULL << 53;
        auto [s1, E1, f1] = Unpack(left);
        auto [s2, E2, f2] = Unpack(right);

        uint64_t M1 = f1 | hidden;
        uint64_t M2 = f2 | hidden;

        uint64_t E;
        uint64_t M;

        if (E1 > E2) {
            uint64_t shift = E1 - E2;
            M = M1 + (shift >= 64 ? 0 : (M2 >> shift));
            E = E1;
        } else if (E2 > E1) {
            uint64_t shift = E2 - E1;
            M = (shift >= 64 ? 0 : (M1 >> shift)) + M2;
            E = E2;
        } else {
            M = M1 + M2;
            E = E1;
        }
        if (M & carry) {
            M >>= 1;
            ++E;
        }
        return Pack(s1, E, M);
    }

    static std::tuple<uint64_t, uint64_t, uint64_t> Unpack(uint64_t value) {
        uint64_t sign = (value >> 63) & 1ull;
        uint64_t exp = (value >> 52) & 0x7FFull;
        uint64_t frac = value & ((1ull << 52) - 1);
        return {sign, exp, frac};
    }

    static uint64_t Pack(uint64_t sign, uint64_t exp, uint64_t frac) {
        return (sign << 63) | (exp << 52) | (frac & ((1ull << 52) - 1));
    }

    uint64_t Value;
};