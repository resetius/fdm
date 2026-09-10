#pragma once

#include <complex>
#include <map>
#include <utility>
#include <vector>

#include "ns_cyl_spectral_modes.h"

namespace fdm {

template<typename T>
std::vector<std::vector<std::complex<T>>> ns_cyl_block_multipliers(
    const NSCylSpectralModeSet<T>& modes) {
    std::map<std::pair<int, int>, std::vector<NSCylSpectralMode<T>>> blocks;
    for (const auto& mode : modes.modes()) {
        blocks[{mode.m, mode.l}].push_back(mode);
    }

    std::vector<std::vector<std::complex<T>>> result;
    result.reserve(blocks.size());
    for (const auto& entry : blocks) {
        std::vector<std::complex<T>> multipliers;
        for (const auto& mode : entry.second) {
            for (int column = 0; column < mode.column_count; ++column) {
                multipliers.push_back(mode.multiplier);
            }
        }
        result.push_back(std::move(multipliers));
    }
    return result;
}

// Memoized realization of the stable-manifold gluing recurrence.  Method
// supplies zero(), S(), Pminus(), Pplus(), and PplusLinv() on equal-sized
// real packed vectors.
template<typename Method>
class NSCylNonlinearGluing {
public:
    using T = typename Method::value_type;
    using Vector = std::vector<T>;

    explicit NSCylNonlinearGluing(Method& method) : method_(method) { }

    Vector operator()(const Vector& y, int iterations, int level = 0) {
        const auto key = std::make_pair(level, iterations);
        const auto known = memo_.find(key);
        if (known != memo_.end()) {
            return known->second;
        }
        if (iterations <= 0) {
            return method_.zero();
        }

        const Vector first = (*this)(y, iterations-1, level);
        const Vector image = method_.S(add(first, y));
        const Vector stable_image = method_.Pminus(image);
        const Vector unstable_image = method_.Pplus(image);
        const Vector continuation =
            (*this)(stable_image, iterations-1, level+1);
        Vector result = add(
            method_.PplusLinv(subtract(continuation, unstable_image)),
            first);
        memo_[key] = result;
        return result;
    }

    void clear() { memo_.clear(); }

private:
    static Vector add(const Vector& first, const Vector& second) {
        Vector result(first.size());
        for (std::size_t index = 0; index < first.size(); ++index) {
            result[index] = first[index]+second[index];
        }
        return result;
    }

    static Vector subtract(const Vector& first, const Vector& second) {
        Vector result(first.size());
        for (std::size_t index = 0; index < first.size(); ++index) {
            result[index] = first[index]-second[index];
        }
        return result;
    }

    Method& method_;
    std::map<std::pair<int, int>, Vector> memo_;
};

} // namespace fdm
