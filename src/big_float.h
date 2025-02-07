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

#include <array>
#include <limits>
#include <string>
#include <iostream>

template<int blocks>
class BigFloat {
public:
    static_assert(blocks > 1, "blocks must be greater than 1");

    BigFloat() = default;

    BigFloat(double number)
        : BigFloat(FromDouble(number))
    { }

    BigFloat(const std::string& str)
        : BigFloat(FromString(str))
    { }

    //operator std::string() const {
    //    return ToString();
    //}

    //operator double() const {
    //    return ToDouble();
    //}

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

    static BigFloat FromDouble(double number) {
        BigFloat result;
        union {
            double d;
            uint64_t u;
        } val;

        val.d = number;

        if (val.u == 0) {
            return {};
        }

        uint64_t bits = val.u;
        result.sign = (bits >> 63) & 0x1;

        int exponent_raw = (bits >> 52) & 0x7FF;
        int exponent = exponent_raw - 1023;
        uint64_t mantissa = bits & 0xFFFFFFFFFFFFFULL;

        mantissa |= (1ULL << 52);
        mantissa <<= 63-52;

        result.mantissa[blocks-1] = static_cast<uint32_t>(mantissa >> 32);
        result.mantissa[blocks-2] = static_cast<uint32_t>(mantissa);

        result.exponent = exponent - (blocks*32-1);

        return result;
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

        int exponent_raw = exponent + (blocks*32-1) + 1023;

        val.u |= static_cast<uint64_t>(exponent_raw) << 52;
        uint64_t mantissa_raw = mantissa[blocks-1];
        mantissa_raw <<= 32;
        mantissa_raw |= mantissa[blocks-2];
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

        if (exponent >= - (blocks * 32 - 1)) {
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
            uint64_t carry = 0;

            if (blocks * 32 - std::abs(exponent) > 0) {
                shiftMantissaLeft(value, blocks * 32 - std::abs(exponent));
            }

            int32_t effectiveExp = exponent + (blocks*32-1);

            while (value != std::array<uint32_t,blocks>{} && fracPart.size() < 20) {
                carry = 0;
                for (size_t i = 0; i < value.size(); i++) {
                    uint64_t current = static_cast<uint64_t>(value[i]) * 10ULL + carry;
                    value[i] = static_cast<uint32_t>(current);
                    carry = current >> 32;
                }

                for (int i = 0; i < 4 && effectiveExp < -1; ++i) {
                    shiftMantissaRight(value);
                    value.back() |= (carry & 1) << 31;
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

    BigFloat operator+(const BigFloat& other) {
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

        uint64_t carry = 0;
        for (size_t i = 0; i < blocks; ++i) {
            uint64_t sum = static_cast<uint64_t>(a.mantissa[i]) +
                        static_cast<uint64_t>(b.mantissa[i]) + carry;
            result.mantissa[i] = static_cast<uint32_t>(sum);
            carry = sum >> 32;
        }

        result.exponent = a.exponent;
        result.sign = sign;

        if (carry) {
            shiftMantissaRight(result.mantissa);
            result.mantissa[blocks-1] |= (carry << 31);
            result.exponent++;
        }

        result.normalize();
        return result;
    }

    BigFloat operator-(const BigFloat& other) {
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

        int64_t borrow = 0;
        for (size_t i = 0; i < blocks; ++i) {
            int64_t diff = static_cast<int64_t>(a.mantissa[i]) -
                        static_cast<int64_t>(b.mantissa[i]) - borrow;
            if (diff < 0) {
                diff += (1ULL << 32);
                borrow = 1;
            } else {
                borrow = 0;
            }
            result.mantissa[i] = static_cast<uint32_t>(diff);
        }

        result.exponent = a.exponent;
        result.sign = swapped ? !sign : sign;

        result.normalize();
        return result;
    }

    BigFloat operator*(const BigFloat& other) {
        if (IsZero() || other.IsZero()) {
            return BigFloat();
        }

        BigFloat result;

        result.exponent = exponent + other.exponent + 1;

        std::array<uint32_t, blocks * 2> temp{0};

        for (size_t i = 0; i < blocks; ++i) {
            uint64_t carry = 0;

            for (size_t j = 0; j < blocks; ++j) {
                size_t pos = i + j;

                uint64_t prod = static_cast<uint64_t>(mantissa[i]) *
                            static_cast<uint64_t>(other.mantissa[j]) +
                            static_cast<uint64_t>(temp[pos]) +
                            carry;

                temp[pos] = static_cast<uint32_t>(prod);

                carry = prod >> 32;
            }

            if (carry && i + blocks < temp.size()) {
                temp[i + blocks] = static_cast<uint32_t>(carry);
            }
        }

        for (size_t i = 0; i < blocks; ++i) {
            result.mantissa[i] = temp[i + blocks];
        }

        result.exponent += blocks * 32 - 1;
        result.normalize();
        result.sign = sign != other.sign;
        return result;
    }

    bool IsZero() const {
        return mantissa == std::array<uint32_t, blocks>{0};
    }

private:
    std::array<uint32_t, blocks> mantissa = {0};
    int32_t exponent = 0;
    bool sign = false;

    static BigFloat IntFromString(const std::string& intPart) {
        BigFloat result;

        uint64_t value = std::stoll(intPart);
        // TODO: handle overflow

        result.mantissa[0] = static_cast<uint32_t>(value & 0xFFFFFFFF);
        result.mantissa[1] = static_cast<uint32_t>(value >> 32);

        result.normalize();
        return result;
    }

    static BigFloat FracFromString(const std::string& fracPart) {
        BigFloat result;

        int mult = 10;
        result.exponent = - (int)blocks*32;
        for (size_t i = 0; i < fracPart.size()-1; i++) {
            mult *= 10;
        }
        // todo:
        uint64_t frac = std::stoll(fracPart) & (0xffffffffU);
        int blockId = blocks-1;
        int bitId = 31;
        frac *= 2;
        while (frac != 0) {
            uint32_t bit = frac >= mult;
            if (bit) {
                frac -= mult;
            }
            frac = 2 * frac;
            result.mantissa[blockId] |= (bit << bitId);
            bitId -= 1;
            if (bitId < 0) {
                bitId = 31;
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
        if (isNormalized()) {
            return;
        }
        int shift = 0;
        for (int i = blocks - 1; i >= 0; --i) {
            if (mantissa[i] == 0) {
                shift += 32;
            } else {
                shift += __builtin_clz(mantissa[i]);
                break;
            }
        }

        exponent -= shift;
        shiftMantissaLeft(mantissa, shift);
    }

    bool isNormalized() const {
        return IsZero() || (mantissa.back() & (1U << 31)) != 0;
    }

    void shiftMantissaLeft(std::array<uint32_t, blocks>& mantissa, int shift = 1) const {
        uint32_t carry = 0;
        int blockShift = shift / 32;
        int bitShift = shift % 32;

        if (blockShift > 0) {
            // blockShift = 1
            // [0, 1, 2, 3, 4] -> [1, 2, 3, 4, 0]
            //  ^  ^  ^  ^  ^
            //  4  3  2  1  0
            int i = blocks - 1;
            for (i = blocks - 1; i >= blockShift; --i) {
                mantissa[i] = mantissa[i - blockShift];
            }
            for (; i >= 0; --i) {
                mantissa[i] = 0;
            }
        }

        uint32_t mask = (1U << bitShift) - 1;

        for (size_t i = blockShift; i < blocks; ++i) {
            uint32_t next_carry = (mantissa[i] & (mask << (32 - bitShift))) >> (32 - bitShift);
            mantissa[i] = (mantissa[i] << bitShift) | carry;
            carry = next_carry;
        }
    }

    void shiftMantissaRight(std::array<uint32_t, blocks>& mantissa, int shift = 1) const {
        uint32_t carry = 0;
        int blockShift = shift / 32;
        int bitShift = shift % 32;

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

        uint32_t mask = (1U << bitShift) - 1;

        for (int i = blocks - blockShift - 1; i >= 0; --i) {
            uint32_t next_carry = mantissa[i] & mask;
            mantissa[i] = (mantissa[i] >> bitShift) | (carry << (32 - bitShift));
            carry = next_carry;
        }
    }
};

namespace std {

template<int blocks>
inline bool isnan(const ::BigFloat<blocks>& a) noexcept {
    return false;
}

template<int blocks>
inline bool isinf(const ::BigFloat<blocks>& a) noexcept {
    return false;
}

} // std
