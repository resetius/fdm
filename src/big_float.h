#pragma once

/*
BSD 3-Clause License

Copyright (c) 2025, Alexey Ozeritskiy
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
https://github.com/resetius/fdm/blob/master/src/big_float.h
*/

/*
M1 benchmark for 32bit block size

sum BigFloat<2> : 0.0257841
sum BigFloat<4> : 0.0213832
sum BigFloat<8> : 0.0296356
sum BigFloat<16> : 0.0519466
sum float : 4.2e-08
sum double : 1.67e-07
mul BigFloat<2> : 0.0107144
mul BigFloat<4> : 0.0233051
mul BigFloat<8> : 0.0918952
mul BigFloat<16> : 0.183747
mul float : 0
mul double : 0

64bit block size
sum BigFloat<2> : 0.0190474
sum BigFloat<4> : 0.0202527
sum BigFloat<8> : 0.0277267
sum BigFloat<16> : 0.0608131
sum float : 0
sum double : 4.2e-08
mul BigFloat<2> : 0.0107325
mul BigFloat<4> : 0.0219723
mul BigFloat<8> : 0.0752863
mul BigFloat<16> : 0.210909
mul float : 0
mul double : 0

*/

#include <array>
#include <limits>
#include <string>
#include <iostream>
#include <cstdint>
#include <algorithm>

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

template<typename BlockType>
struct GenericPlatformSpec {
    static inline unsigned char addcarry(unsigned char carry, BlockType a, BlockType b, BlockType* sum)
    {
        bool overflow1 = __builtin_add_overflow(a, carry, sum);
        bool overflow2 = __builtin_add_overflow(b, *sum, sum);
        return (unsigned char)(overflow1 | overflow2);
    }

    static inline unsigned char addcarry(BlockType a, BlockType b, BlockType* sum)
    {
        bool overflow = __builtin_add_overflow(a, b, sum);
        return (unsigned char)overflow;
    }

    static inline unsigned char subborrow(unsigned char borrow, BlockType a, BlockType b, BlockType* res)
    {
        bool overflow1 = __builtin_add_overflow(b, borrow, &b);
        bool overflow2 = __builtin_sub_overflow(a, b, res);
        return (unsigned char)(overflow1 | overflow2);
    }

    static inline void umul_ppmm(BlockType* hi, BlockType* lo, BlockType a, BlockType b) {
        if constexpr (std::is_same_v<BlockType, uint32_t>) {
            uint64_t tmp = static_cast<uint64_t>(a) * b;
            *lo = static_cast<uint32_t>(tmp);
            *hi = static_cast<uint32_t>(tmp >> 32);
        } else if constexpr (std::is_same_v<BlockType, uint64_t>) {
            const uint64_t a0 = (uint32_t)a, a1 = a >> 32;
            const uint64_t b0 = (uint32_t)b, b1 = b >> 32;
            const uint64_t p11 = a1 * b1, p01 = a0 * b1;
            const uint64_t p10 = a1 * b0, p00 = a0 * b0;

            const uint64_t middle = p10 + (p00 >> 32) + (uint32_t)p01;

            *hi = p11 + (middle >> 32) + (p01 >> 32);
            *lo = (middle << 32) | (uint32_t)p00;
        } else {
            static_assert(false, "Unsupported BlockType");
        }
    }
};


#if defined(__x86_64__)
template<typename BlockType>
struct AMD64PlatformSpec {
    static inline unsigned char addcarry(unsigned char carry, BlockType a, BlockType b, BlockType* sum) {
        if constexpr (std::is_same_v<BlockType, uint32_t>) {
            return _addcarry_u32(carry, a, b, (unsigned int*)sum);
        }
        else if constexpr (std::is_same_v<BlockType, uint64_t>) {
            return _addcarry_u64(carry, a, b, (unsigned long long*)sum);
        }
        else {
            static_assert(false, "Unsupported BlockType");
        }
    }

    static inline unsigned char addcarry(BlockType a, BlockType b, BlockType* sum) {
        return addcarry(0, a, b, sum);
    }

    static inline unsigned char subborrow(unsigned char borrow, BlockType a, BlockType b, BlockType* res) {
        if constexpr (std::is_same_v<BlockType, uint32_t>) {
            return _subborrow_u32(borrow, a, b, (unsigned int*)res);
        } else if constexpr (std::is_same_v<BlockType, uint64_t>) {
            return _subborrow_u64(borrow, a, b, (unsigned long long*)res);
        } else {
            static_assert(false, "Unsupported BlockType");
        }
    }

    static inline void umul_ppmm(BlockType* hi, BlockType* lo, BlockType a, BlockType b) {
        if constexpr (std::is_same_v<BlockType, uint32_t>) {
            GenericPlatformSpec<BlockType>::umul_ppmm(hi, lo, a, b);
        } else if constexpr (std::is_same_v<BlockType, uint64_t>) {
            asm volatile(
                "mulx %[b], %[lo], %[hi]"
                : [lo] "=r" (*lo), [hi] "=r" (*hi)
                : "d" (a), [b] "r" (b)
            );
        } else {
            static_assert(false, "Unsupported BlockType");
        }
    }
};
#endif


#if defined(__aarch64__)
template<typename BlockType>
struct AArch64PlatformSpec {
    static inline unsigned char addcarry(unsigned char carry, BlockType a, BlockType b, BlockType* sum) {
        if constexpr (std::is_same_v<BlockType, uint32_t>) {
            return GenericPlatformSpec<BlockType>(carry, a, b, sum);
        } else if constexpr (std::is_same_v<BlockType, uint64_t>) {
            unsigned char carry_out;
            uint64_t result;

            asm volatile (
                "adds xzr, xzr, xzr\n\t"
                "adcs %[result], %[a], %[b]\n\t"
                "cset %[carry_out], cs"
                : [result] "=r" (result), [carry_out] "=r" (carry_out)
                : [a] "r" (a), [b] "r" (b)
                : "cc"
            );

            *sum = result;
            return carry_out;
        } else {
            static_assert(false, "Unsupported BlockType");
        }
    }

    static inline unsigned char addcarry(BlockType a, BlockType b, BlockType* sum) {
        if constexpr (std::is_same_v<BlockType, uint32_t>) {
            return GenericPlatformSpec<BlockType>::addcarry(a, b, sum);
        } else if constexpr (std::is_same_v<BlockType, uint64_t>) {
            unsigned char carry_out;
            uint64_t result;

            asm volatile (
                "adds %[result], %[a], %[b]\n\t"
                "cset %[carry_out], cs"
                : [result] "=r" (result), [carry_out] "=r" (carry_out)
                : [a] "r" (a), [b] "r" (b)
                : "cc"
            );

            *sum = result;
            return carry_out;
        } else {
            static_assert(false, "Unsupported BlockType");
        }
    }

    static inline unsigned char subborrow(unsigned char borrow, BlockType a, BlockType b, BlockType* diff)
    {
        if constexpr (std::is_same_v<BlockType, uint32_t>) {
            return GenericPlatformSpec<BlockType>::subborrow(borrow, a, b, diff);
        } else if constexpr (std::is_same_v<BlockType, uint64_t>) {
            uint64_t res;
            unsigned int outBorrow;
            asm volatile(
                "mov    w10, #1           \n"
                "sub    w10, w10, %w[br]   \n"
                "cmp    w10, #1           \n"
                "sbcs   %x[res], %x[a], %x[b] \n"
                "cset   %w[ob], cc        \n"
                : [res] "=&r" (res), [ob] "=&r" (outBorrow)
                : [a] "r" (a), [b] "r" (b), [br] "r" ((unsigned int)borrow)
                : "w10", "cc"
            );
            *diff = res;
            return (unsigned char) outBorrow;
        } else {
            static_assert(false, "Unsupported BlockType");
        }
    }


    static inline void umul_ppmm(BlockType* hi, BlockType* lo, BlockType a, BlockType b) {
        if constexpr (std::is_same_v<BlockType, uint32_t>) {
            GenericPlatformSpec<BlockType>::umul_ppmm(hi, lo, a, b);
        } else if constexpr (std::is_same_v<BlockType, uint64_t>) {
            asm volatile (
                "mul  %0, %2, %3\n\t"
                "umulh %1, %2, %3"
                : "=&r"(*lo), "=&r"(*hi)
                : "r"(a), "r"(b)
            );
        } else {
            static_assert(false, "Unsupported BlockType");
        }
    }
};
#endif

#if defined(__x86_64__)
template<typename BlockType>
using DefaultPlatformSpec = AMD64PlatformSpec<BlockType>;
#elif defined(__aarch64__)
template<typename BlockType>
using DefaultPlatformSpec = AArch64PlatformSpec<BlockType>;
#else
template<typename BlockType>
using DefaultPlatformSpec = GenericPlatformSpec<BlockType>;
#endif

namespace detail {

static inline int clz(uint32_t a) {
    return __builtin_clz(a);
}

static inline int clz(uint64_t a) {
    return __builtin_clzll(a);
}

} // detail

template<int blocks, typename BlockType = uint64_t, typename Spec = DefaultPlatformSpec<BlockType>>
class BigFloat {
public:
    BigFloat() = default;

    BigFloat(double number)
    {
        union {
            double d;
            uint64_t u;
        } val;

        val.d = number;

        if (val.u == 0) {
            return;
        }

        uint64_t bits = val.u;
        sign = ((bits >> 63) & 0x1) == 0 ? 1 : -1;

        int exponent_raw = (bits >> 52) & 0x7FF;
        exponent = exponent_raw - 1023;
        uint64_t mantissa_value = bits & 0xFFFFFFFFFFFFFULL;

        mantissa_value |= (1ULL << 52);
        mantissa_value <<= 63-52;

        if constexpr(std::is_same_v<BlockType, uint64_t>) {
            mantissa[blocks-1] = mantissa_value;
        } else {
            mantissa[blocks-1] = static_cast<uint32_t>(mantissa_value >> 32);
            mantissa[blocks-2] = static_cast<uint32_t>(mantissa_value);
        }

        exponent = exponent - (blocks*blockBits-1);
    }

    BigFloat(long long number)
    {
        if (number == 0) {
            return;
        }
        sign = (number < 0) ? -1 : 1;
        uint64_t value = std::abs(number);
        if constexpr(std::is_same_v<BlockType, uint64_t>) {
            mantissa[blocks-1] = value;
            exponent = exponent - ((blocks-1)*blockBits);
        } else {
            mantissa[blocks-1] = static_cast<uint32_t>(value >> 32);
            mantissa[blocks-2] = static_cast<uint32_t>(value);
            exponent = exponent - ((blocks-2)*blockBits);
        }
        normalize();
    }

    BigFloat(int number)
        : BigFloat((long long)number)
    { }

    BigFloat(long number)
        : BigFloat((long long)number)
    { }

    template<int otherBlocks, typename OtherSpec>
    BigFloat(const BigFloat<otherBlocks, BlockType, OtherSpec>& other)
        : exponent(other.getExponent())
        , sign(other.getSign())
    {
        auto dst = mantissa.rbegin();
        auto src = other.getMantissa().rbegin();
        while (dst != mantissa.rend() && src != other.getMantissa().rend()) {
            *dst++ = *src++;
        }

        auto diff = (int)std::distance(dst, mantissa.rend()) - (int)std::distance(src, other.getMantissa().rend());
        exponent -= diff * blockBits;
    }

    BigFloat(const std::string& str)
        : BigFloat(FromString(str))
    { }

    explicit operator std::string() const {
        return ToString();
    }

    explicit operator double() const {
        return ToDouble();
    }

    BigFloat operator-() const {
        BigFloat result = *this;
        result.sign = -result.sign;
        return result;
    }

    bool operator<(const BigFloat& other) const {
        if (IsZero()) {
            return other.sign == 1;
        }

        if (other.IsZero()) {
            return sign == -1;
        }

        if (sign != other.sign) {
            return sign < other.sign;
        }

        if (sign > 0) {
            if (exponent != other.exponent) {
                return exponent < other.exponent;
            }

            return less(mantissa, other.mantissa);
        } else {
            if (exponent != other.exponent) {
                return other.exponent < exponent;
            }

            return less(other.mantissa, mantissa);
        }
    }

    bool operator>(const BigFloat& other) const {
        return other < *this;
    }

    bool operator==(const BigFloat& other) const {
        return sign == other.sign && exponent == other.exponent && mantissa == other.mantissa;
    }

    double ToDouble() const {
        if (IsZero()) {
            return 0.0;
        }

        union {
            double d;
            uint64_t u;
        } val;

        val.u = 0;

        int exponent_raw = exponent + (blocks*blockBits-1) + 1023;

        val.u |= static_cast<uint64_t>(exponent_raw) << 52;
        uint64_t mantissa_raw;
        if constexpr(std::is_same_v<BlockType,uint32_t>) {
            mantissa_raw = mantissa[blocks-1];
            mantissa_raw <<= 32;
            mantissa_raw |= mantissa[blocks-2];
        } else {
            mantissa_raw = mantissa[blocks-1];
        }
        mantissa_raw >>= 63-52;
        val.u |= mantissa_raw & 0xFFFFFFFFFFFFFULL;
        val.u |= sign == -1 ? static_cast<uint64_t>(1) << 63 : 0;

        return val.d;
    }

    static BigFloat FromString(const std::string& str) {
        BigFloat result;

        size_t pos = 0;
        int sign = 1;
        if (str[0] == '-') {
            sign = -1;
            pos++;
        }

        std::string intPart = "";
        std::string fracPart = "";

        while (pos < str.length() && str[pos] != '.') {
            intPart += str[pos++];
        }

        if (pos < str.length() && str[pos] == '.') {
            pos++;
            while (pos < str.length()) {
                fracPart += str[pos++];
            }
        }

        if (!intPart.empty()) {
            result = IntFromString(intPart);
        }

        if (!fracPart.empty()) {
            auto frac = FracFromString(fracPart);
            result = result + frac;
        }

        if (!result.IsZero()) {
            result.sign = sign;
        }

        return result;
    }

    std::string ToString() const {
        std::string result;

        if (sign == -1) {
            result += "-";
        }

        std::string intPart = "";
        std::string fracPart = "";

        if (exponent >= - (blocks * blockBits - 1)) {
            // integer part
            auto value = mantissa;
            shiftMantissaRight(value, std::abs(exponent));
            int blockId = blocks - 1;
            for (; blockId >= 0; --blockId) {
                if (value[blockId] != 0) {
                    break;
                }
            }
            for (int i = 0; i <= blockId; i++) {
                while (value[i] != 0) {
                    auto r = value[i] % 10;
                    value[i] /= 10;
                    intPart += r + '0';
                    // todo: leading zeros
                }
            }
            std::reverse(intPart.begin(), intPart.end());
        }

        if (1) {
            auto value = mantissa;
            WideType carry = 0;

            if (blocks * blockBits - std::abs(exponent) > 0) {
                shiftMantissaLeft(value, blocks * blockBits - std::abs(exponent));
            }

            int32_t effectiveExp = exponent + (blocks*blockBits-1);

            while (!IsZero(value) && fracPart.size() < 18) {
                carry = 0;

                for (size_t i = 0; i < value.size(); i++) {
                    WideType current = static_cast<WideType>(value[i]) * 10ULL + carry;
                    value[i] = static_cast<BlockType>(current);
                    carry = current >> blockBits;
                }

                for (int i = 0; i < 4 && effectiveExp < -1; ++i) {
                    shiftMantissaRight(value);
                    value.back() |= (carry & 1) << (blockBits - 1);
                    effectiveExp++;
                    carry >>= 1;
                }
                fracPart += carry + '0';
            }
        }

        if (intPart == "") {
            intPart = "0";
        }

        result += intPart;
        if (fracPart != "") {
            result += ".";
            result += fracPart;
        }
        return result;
    }

    BigFloat& operator+=(const BigFloat& other) {
        if (other.IsZero()) {
            return *this;
        }
        if (IsZero()) {
            return (*this = other);
        }

        if (sign != other.sign) {
            BigFloat temp = other;
            temp.sign = -temp.sign;
            return SubUnchecked(temp);
        }

        return AddUnchecked(other);
    }

    BigFloat operator+(const BigFloat& other) const {
        BigFloat result = *this;
        return (result += other);
    }

    BigFloat& operator-=(const BigFloat& other) {
        if (other.IsZero()) {
            return *this;
        }
        if (IsZero()) {
            return (*this = other);
        }

        if (sign != other.sign) {
            BigFloat temp = other;
            temp.sign = -temp.sign;
            return AddUnchecked(temp);
        }

        return SubUnchecked(other);
    }

    BigFloat operator-(const BigFloat& other) const {
        BigFloat result = *this;
        return (result -= other);
    }

    BigFloat Mul2() const {
        BigFloat result = *this;
        result.exponent++;
        return result;
    }

    BigFloat operator*(const BigFloat& other) const {
        if (IsZero() || other.IsZero()) {
            return BigFloat();
        }

        BigFloat result;

        result.exponent = exponent + other.exponent + blocks * blockBits;

        std::array<BlockType, blocks * 2> temp{0};
        mulWithCarry(temp, mantissa, other.mantissa);

        normalize(temp, result.exponent);
        for (size_t i = 0; i < blocks; ++i) {
            result.mantissa[i] = temp[i + blocks];
        }
        result.sign = sign * other.sign;
        return result;
    }

    int getSign() const {
        return sign;
    }

    int getExponent() const {
        return exponent;
    }

    auto& getMantissa() const {
        return mantissa;
    }

private:
    using WideType = std::conditional_t<std::is_same_v<BlockType, uint64_t>, unsigned __int128, uint64_t>;
    using SignedWideType = std::conditional_t<std::is_same_v<BlockType, uint64_t>, __int128, int64_t>;

    std::array<BlockType, blocks> mantissa = {0};
    int32_t exponent = 0;
    int sign = 0; // -1 negitive, 0 zero, 1 positive
    static constexpr int blockBits = sizeof(BlockType) * 8;
    static_assert(std::is_same_v<BlockType,uint64_t> || blocks > 1, "blocks must be greater than 1");
    static_assert(std::is_same_v<BlockType,uint64_t> || std::is_same_v<BlockType,uint32_t>);

    BigFloat& AddUnchecked(const BigFloat& other) {
        auto exp_diff = exponent - other.exponent;
        BlockType carry = 0;

        if (exp_diff > 0) {
            BigFloat b = other;
            shiftMantissaRight(b.mantissa, exp_diff);
            carry = sumWithCarry(mantissa, mantissa, b.mantissa);
        } else if (exp_diff < 0) {
            shiftMantissaRight(mantissa, -exp_diff);
            exponent -= exp_diff;
            carry = sumWithCarry(mantissa, mantissa, other.mantissa);
        } else {
            carry = sumWithCarry(mantissa, mantissa, other.mantissa);
        }

        if (carry) {
            shiftMantissaRight(mantissa);
            mantissa[blocks-1] |= carry << (blockBits - 1);
            exponent++;
        }

        normalize();
        return *this;
    }

    BigFloat& SubUnchecked(const BigFloat& other) {
        auto exp_diff = exponent - other.exponent;

        auto sub = [&](const BigFloat* a, const BigFloat* b) {
            if (less(a->mantissa, b->mantissa)) {
                std::swap(a, b);
                sign = -sign;
            }
            subWithBorrow(mantissa, a->mantissa, b->mantissa);
            exponent = a->exponent;
        };

        if (exp_diff > 0) {
            BigFloat b = other;
            shiftMantissaRight(b.mantissa, exp_diff);
            b.exponent += exp_diff;
            sub(this, &b);
        } else if (exp_diff < 0) {
            shiftMantissaRight(mantissa, -exp_diff);
            exponent -= exp_diff;
            sub(this, &other);
        } else {
            sub(this, &other);
        }

        normalize();
        return *this;
    }

    bool IsZero() const {
        return sign == 0;
    }

    template<size_t array_blocks>
    static bool IsZero(const std::array<BlockType, array_blocks>& array) {
        for (int i = (int)array_blocks - 1; i >= 0; --i) {
            if (array[i] != 0) {
                return false;
            }
        }
        return true;
    }

    static BigFloat IntFromString(const std::string& intPart) {
        BigFloat result;

        uint64_t value = std::stoll(intPart);
        if (value == 0) {
            return {};
        }
        result.sign = 1;
        // TODO: handle overflow

        if constexpr(std::is_same_v<BlockType,uint32_t>) {
            result.mantissa[0] = static_cast<uint32_t>(value & 0xFFFFFFFF);
            result.mantissa[1] = static_cast<uint32_t>(value >> 32);
        } else {
            result.mantissa[0] = static_cast<uint32_t>(value);
        }

        result.normalize();
        return result;
    }

    static BigFloat FracFromString(const std::string& fracPart) {
        BigFloat result;

        WideType mult = 10;
        result.exponent = - blocks*blockBits;
        for (size_t i = 0; i < fracPart.size()-1; i++) {
            mult *= 10;
        }
        // todo:
        WideType frac = std::stoll(fracPart);
        int blockId = blocks-1;
        int bitId = blockBits-1;
        frac *= 2;
        while (frac != 0) {
            BlockType bit = frac >= mult;
            if (bit) {
                frac -= mult;
            }
            frac = 2 * frac;
            result.mantissa[blockId] |= (bit << bitId);
            bitId -= 1;
            if (bitId < 0) {
                bitId = blockBits - 1;
                blockId --;
            }
            if (blockId < 0) {
                break;
            }
        }

        if (IsZero(result.mantissa)) {
            return {};
        }

        result.sign = 1;
        result.normalize();

        return result;
    }

    static bool less(const std::array<BlockType, blocks>& left, const std::array<BlockType, blocks>& right) {
        for (int i = blocks-1; i >= 0; --i) {
            if (left[i] != right[i]) {
                return left[i] < right[i];
            }
        }
        return false;
    }

    void normalize() {
        normalize(mantissa, exponent);
    }

    template<size_t array_blocks>
    static void normalize(std::array<BlockType, array_blocks>& array, int& exp) {
        if (isNormalized(array)) {
            return;
        }
        int shift = 0;
        for (int i = array_blocks - 1; i >= 0; --i) {
            if (array[i] == 0) {
                shift += blockBits;
            } else {
                shift += detail::clz(array[i]);
                break;
            }
        }

        exp -= shift;
        shiftMantissaLeft(array, shift);
    }

    static BlockType sumWithCarry(
        std::array<BlockType, blocks>& result,
        const std::array<BlockType, blocks>& a,
        const std::array<BlockType, blocks>& b)
    {
        BlockType carry = Spec::addcarry(a[0], b[0], &result[0]);
        for (size_t i = 1; i < blocks; ++i) {
            carry = Spec::addcarry(carry, a[i], b[i], &result[i]);
        }
        return carry;
    }

    static void subWithBorrow(
        std::array<BlockType, blocks>& result,
        const std::array<BlockType, blocks>& a,
        const std::array<BlockType, blocks>& b)
    {
        BlockType borrow = 0;
        for (size_t i = 0; i < blocks; ++i) {
            borrow = Spec::subborrow(borrow, a[i], b[i], &result[i]);
        }
    }

    static void mulWithCarry(
        std::array<BlockType, 2*blocks>& result,
        const std::array<BlockType, blocks>& a,
        const std::array<BlockType, blocks>& b)
    {
        for (size_t i = 0; i < blocks; ++i)
        {
            BlockType carry = 0;

            for (size_t j = 0; j < blocks; ++j) {
                size_t pos = i + j;

                BlockType hi, lo;
                Spec::umul_ppmm(&hi, &lo, a[i], b[j]);
                hi += __builtin_add_overflow(lo, result[pos], &lo);
                carry = __builtin_add_overflow(lo, carry, &lo) + hi;
                result[pos] = lo;
            }

            result[i + blocks] += carry;
        }
    }

    bool isNormalized() const {
        return IsZero() || isNormalized(mantissa);
    }

    template<size_t array_blocks>
    static bool isNormalized(const std::array<BlockType, array_blocks>& array) {
        return (array.back() & ((BlockType)1U << (blockBits-1))) != 0;
    }

    template<size_t array_blocks>
    static void shiftMantissaLeft(std::array<BlockType, array_blocks>& mantissa, int shift = 1) {
        BlockType carry = 0;
        int blockShift = shift / blockBits;
        int bitShift = shift % blockBits;

        if (blockShift > 0) {
            // blockShift = 1
            // [0, 1, 2, 3, 4] -> [1, 2, 3, 4, 0]
            //  ^  ^  ^  ^  ^
            //  4  3  2  1  0
            int i = array_blocks - 1;
            for (i = array_blocks - 1; i >= blockShift; --i) {
                mantissa[i] = mantissa[i - blockShift];
            }
            for (; i >= 0; --i) {
                mantissa[i] = 0;
            }
        }

        if (bitShift > 0) {
            BlockType mask = (1ULL << bitShift) - 1ULL;
            for (size_t i = blockShift; i < array_blocks; ++i) {
                BlockType next_carry = (mantissa[i] & (mask << (blockBits - bitShift))) >> (blockBits - bitShift);
                mantissa[i] = (mantissa[i] << bitShift) | carry;
                carry = next_carry;
            }
        }
    }

    static void shiftMantissaRight(std::array<BlockType, blocks>& mantissa, int shift = 1) {
        BlockType carry = 0;
        int blockShift = shift / blockBits;
        int bitShift = shift % blockBits;

        if (blockShift > 0) {
            // blockShift = 1
            // [0, 1, 2, 3, 4] -> [0, 0, 1, 2, 3]
            //  ^  ^  ^  ^  ^
            //  4  3  2  1  0
            int i = 0;
            for (i = 0; i < blocks - blockShift; ++i) {
                mantissa[i] = mantissa[i + blockShift];
            }
            for (; i < blocks; ++i) {
                mantissa[i] = 0;
            }
        }

        if (bitShift > 0) {
            BlockType mask = (1ULL << bitShift) - 1ULL;
            int i;
            for (i = 0; i < blocks - blockShift - 1; i++) {
                carry = mantissa[i+1] & mask;
                if constexpr(std::is_same_v<BlockType,uint64_t>) {
//#ifdef __x86_64__
//                    asm volatile ("shrd %2, %1, %0"
//                        : "+r" (mantissa[i])
//                        : "r" (carry), "c" ((unsigned char)bitShift)
//                    );
//#else
                    mantissa[i] = (mantissa[i] >> bitShift) | (carry << (blockBits - bitShift));
//#endif
                } else {
                    mantissa[i] = (mantissa[i] >> bitShift) | (carry << (blockBits - bitShift));
                }
            }
            mantissa[i] >>= bitShift;
        }
    }

    friend void test_precision(void**);
};
