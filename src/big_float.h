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

namespace detail {

static inline int clz(uint32_t a) {
    return __builtin_clz(a);
}

static inline int clz(uint64_t a) {
    return __builtin_clzll(a);
}

#if defined(__x86_64__)
static inline unsigned char addcarry_u64(unsigned char carry, uint64_t a, uint64_t b, uint64_t* sum) {
    return _addcarry_u64(carry, a, b, (unsigned long long*)sum);
}

static inline unsigned char addcarry_u64(uint64_t a, uint64_t b, uint64_t* sum) {
    return _addcarry_u64(0, a, b, (unsigned long long*)sum);
}

static inline unsigned char subborrow_u64(unsigned char borrow, uint64_t a, uint64_t b, uint64_t* res) {
    return _subborrow_u64(borrow, a, b, (unsigned long long*)res);
}

#elif defined(__aarch64__)

static inline unsigned char addcarry_u64(unsigned char carry, uint64_t a, uint64_t b, uint64_t* sum) {
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
}

static inline unsigned char addcarry_u64(uint64_t a, uint64_t b, uint64_t* sum) {
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
}

static inline unsigned char subborrow_u64(unsigned char borrow, uint64_t a, uint64_t b, uint64_t* diff) {
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
}

#endif

} // detail

template<int blocks, typename BlockType = uint64_t>
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
        sign = (bits >> 63) & 0x1;

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
        sign = number < 0;
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

    template<int otherBlocks>
    BigFloat(const BigFloat<otherBlocks, BlockType>& other)
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
        result.sign = !result.sign;
        return result;
    }

    bool operator<(const BigFloat& other) const {
        if (IsZero()) {
            return !other.sign;
        }

        if (other.IsZero()) {
            return sign;
        }

        if (sign != other.sign) {
            return sign;
        }

        if (sign) {
            return -other < -*this;
        }

        if (exponent != other.exponent) {
            return exponent < other.exponent;
        }

        for (int i = 0; i < blocks; i++) {
            if (mantissa[i] != other.mantissa[i]) {
                return mantissa[i] < other.mantissa[i];
            }
        }

        return false;
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
        val.u |= static_cast<uint64_t>(sign) << 63;

        return val.d;
    }

    static BigFloat FromString(const std::string& str) {
        BigFloat result;

        size_t pos = 0;
        bool sign = false;
        if (str[0] == '-') {
            sign = true;
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

        result.sign = sign;

        return result;
    }

    std::string ToString() const {
        std::string result;

        if (sign) {
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

            while (value != std::array<BlockType,blocks>{} && fracPart.size() < 18) {
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
        *this = *this + other;
        return *this;
    }

    BigFloat operator+(const BigFloat& other) const {
        if (other.IsZero()) {
            return *this;
        }
        if (IsZero()) {
            return other;
        }

        if (sign != other.sign) {
            BigFloat temp = other;
            temp.sign = !temp.sign;
            return *this - temp;
        }

        BigFloat result;
        auto exp_diff = exponent - other.exponent;

        BigFloat a = *this;
        BigFloat b = other;

        if (exp_diff > 0) {
            shiftMantissaRight(b.mantissa, exp_diff);
            b.exponent += exp_diff;
        } else if (exp_diff < 0) {
            shiftMantissaRight(a.mantissa, -exp_diff);
            a.exponent -= exp_diff;
        }

        auto carry = sumWithCarry(result.mantissa, a.mantissa, b.mantissa);
        result.exponent = a.exponent;
        result.sign = sign;

        if (carry) {
            shiftMantissaRight(result.mantissa);
            result.mantissa[blocks-1] |= carry << (blockBits - 1);
            result.exponent++;
        }

        result.normalize();
        return result;
    }

    BigFloat operator-(const BigFloat& other) const {
        if (other.IsZero()) {
            return *this;
        }
        if (IsZero()) {
            BigFloat result = other;
            result.sign = !result.sign;
            return result;
        }

        if (sign != other.sign) {
            BigFloat temp = other;
            temp.sign = !temp.sign;
            return *this + temp;
        }

        BigFloat result;
        auto exp_diff = exponent - other.exponent;

        BigFloat a = *this;
        BigFloat b = other;

        if (exp_diff > 0) {
            shiftMantissaRight(b.mantissa, exp_diff);
            b.exponent += exp_diff;
        } else if (exp_diff < 0) {
            shiftMantissaRight(a.mantissa, -exp_diff);
            a.exponent -= exp_diff;
        }

        bool swapped = false;
        for (int i = blocks - 1; i >= 0; --i) {
            if (a.mantissa[i] < b.mantissa[i]) {
                std::swap(a, b);
                swapped = true;
                break;
            }
            if (a.mantissa[i] > b.mantissa[i]) {
                break;
            }
        }

        subWithBorrow(result.mantissa, a.mantissa, b.mantissa);

        result.exponent = a.exponent;
        result.sign = swapped ? !sign : sign;

        result.normalize();
        return result;
    }

    BigFloat Mul2() const {
        BigFloat result = *this;
        result.exponent++;
        return result;
    }

    BigFloat Square() const {
        if (IsZero()) {
            return {};
        }

        BigFloat result;

        result.exponent = 2 * exponent + blocks * blockBits;

        std::array<BlockType, blocks * 2> temp{0};
        square(temp, mantissa);

        normalize(temp, result.exponent);
        for (size_t i = 0; i < blocks; ++i) {
            result.mantissa[i] = temp[i + blocks];
        }
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
        result.sign = sign != other.sign;
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
    bool sign = false;
    static constexpr int blockBits = sizeof(BlockType) * 8;
    static_assert(std::is_same_v<BlockType,uint64_t> || blocks > 1, "blocks must be greater than 1");
    static_assert(std::is_same_v<BlockType,uint64_t> || std::is_same_v<BlockType,uint32_t>);

    bool IsZero() const {
        return IsZero(mantissa);
    }

    template<size_t array_blocks>
    static bool IsZero(const std::array<BlockType, array_blocks>& array) {
        for (size_t i = array_blocks - 1; i >= 0; --i) {
            if (array[i] != 0) {
                return false;
            }
        }
        return true;
    }

    static BigFloat IntFromString(const std::string& intPart) {
        BigFloat result;

        uint64_t value = std::stoll(intPart);
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

        result.normalize();

        return result;
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
#if defined(__x86_64__) || defined(__aarch64__)
        constexpr bool use_asm = std::is_same_v<BlockType, uint64_t>;
#else
        constexpr bool use_asm = false;
#endif

        if constexpr(use_asm) {
            BlockType carry = detail::addcarry_u64(a[0], b[0], &result[0]);
            for (size_t i = 1; i < blocks; ++i) {
                carry = detail::addcarry_u64(carry, a[i], b[i], &result[i]);
            }
            return carry;
        } else {
            BlockType carry = 0;
            for (size_t i = 0; i < blocks; ++i) {
                BlockType temp = a[i] + carry;
                bool overflow1 = temp < carry;
                BlockType sum = temp + b[i];
                bool overflow2 = sum < temp;
                result[i] = sum;
                carry = overflow1 | overflow2;
            }
            return carry;
        }
    }

    static void subWithBorrow(
        std::array<BlockType, blocks>& result,
        const std::array<BlockType, blocks>& a,
        const std::array<BlockType, blocks>& b)
    {
#if defined(__x86_64__) || defined(__aarch64__)
        constexpr bool use_asm = std::is_same_v<BlockType, uint64_t>;
#else
        constexpr bool use_asm = false;
#endif

        if constexpr(use_asm) {
            BlockType borrow = 0;
            for (size_t i = 0; i < blocks; ++i) {
                borrow = detail::subborrow_u64(borrow, a[i], b[i], &result[i]);
            }
        } else {
            BlockType borrow = 0;
            for (size_t i = 0; i < blocks; ++i) {
                BlockType subtrahend = b[i] + borrow;
                bool overflow = (subtrahend < b[i]);
                result[i] = a[i] - subtrahend;
                borrow = (a[i] < subtrahend)  | overflow;
            }
        }
    }

    static void mulWithCarry(
        std::array<BlockType, 2*blocks>& result,
        const std::array<BlockType, blocks>& a,
        const std::array<BlockType, blocks>& b)
    {
        for (size_t i = 0; i < blocks; ++i)
        {
            WideType carry = 0;

            for (size_t j = 0; j < blocks; ++j) {
                size_t pos = i + j;

                WideType prod = static_cast<WideType>(a[i]) *
                            static_cast<WideType>(b[j]) +
                            static_cast<WideType>(result[pos]) +
                            carry;

                result[pos] = static_cast<BlockType>(prod);

                carry = prod >> blockBits;
            }

            result[i + blocks] += static_cast<BlockType>(carry);
        }
    }

    static void square(std::array<BlockType, 2*blocks>& result,
                    const std::array<BlockType, blocks>& a)
    {
        for (size_t i = 0; i < blocks; ++i) {
            WideType carry = 0;
            for (size_t j = i + 1; j < blocks; ++j) {
                size_t pos = i + j;

                WideType prod = static_cast<WideType>(a[i]) * static_cast<WideType>(a[j]);
                prod <<= 1;  // prod *= 2;

                WideType sum = static_cast<WideType>(result[pos]) + prod + carry;
                result[pos] = static_cast<BlockType>(sum);
                carry = sum >> blockBits;
            }
            size_t pos = i + blocks;
            while (carry && pos < result.size()) {
                WideType sum = static_cast<WideType>(result[pos]) + carry;
                result[pos] = static_cast<BlockType>(sum);
                carry = sum >> blockBits;
                ++pos;
            }
        }

        for (size_t i = 0; i < blocks; ++i) {
            size_t pos = 2 * i;
            WideType prod = static_cast<WideType>(a[i]) * static_cast<WideType>(a[i]);
            WideType sum = static_cast<WideType>(result[pos]) + prod;
            result[pos] = static_cast<BlockType>(sum);
            WideType carry = sum >> blockBits;

            size_t k = pos + 1;
            while (carry && k < result.size()) {
                sum = static_cast<WideType>(result[k]) + carry;
                result[k] = static_cast<BlockType>(sum);
                carry = sum >> blockBits;
                ++k;
            }
        }
    }

    bool isNormalized() const {
        return isNormalized(mantissa);
    }

    template<size_t array_blocks>
    static bool isNormalized(const std::array<BlockType, array_blocks>& array) {
        return (array.back() & ((BlockType)1U << (blockBits-1))) != 0 || IsZero(array);
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

        BlockType mask = (1ULL << bitShift) - 1ULL;

        for (size_t i = blockShift; i < array_blocks; ++i) {
            BlockType next_carry = (mantissa[i] & (mask << (blockBits - bitShift))) >> (blockBits - bitShift);
            mantissa[i] = (mantissa[i] << bitShift) | carry;
            carry = next_carry;
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

        BlockType mask = (1ULL << bitShift) - 1ULL;

        for (int i = blocks - blockShift - 1; i >= 0; --i) {
            BlockType next_carry = mantissa[i] & mask;
            mantissa[i] = (mantissa[i] >> bitShift) | (carry << (blockBits - bitShift));
            carry = next_carry;
        }
    }

    friend void test_precision(void**);
};
