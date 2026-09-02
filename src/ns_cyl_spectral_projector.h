#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mgsch.h"
#include "ns_cyl_spectral_modes.h"
#include "projection.h"

namespace fdm {

template<typename T>
class NSCylSpectralBlockProjector {
public:
    using Mode = NSCylSpectralMode<T>;

    NSCylSpectralBlockProjector(const std::vector<Mode>& modes,
                                double condition_limit) {
        if (modes.empty()) {
            throw std::invalid_argument("spectral block has no modes");
        }
        if (!(condition_limit >= 1)) {
            throw std::invalid_argument(
                "spectral condition limit must be at least one");
        }

        m_ = modes.front().m;
        l_ = modes.front().l;
        block_size_ = modes.front().block_size;
        phase_count_ = modes.front().phase_count;
        radial_size_ = modes.front().radial_size;
        pressure_gauge_fixed_ = modes.front().pressure_gauge_fixed;
        if (block_size_ <= 0) {
            throw std::invalid_argument("invalid spectral block size");
        }

        for (const auto& mode : modes) {
            validate_metadata(mode);
            append_columns(mode.right_columns, mode.column_count,
                           right_basis_);
            append_columns(mode.left_columns, mode.column_count,
                           left_basis_);
        }
        if (right_basis_.empty() || right_basis_.size() != left_basis_.size()) {
            throw std::invalid_argument("invalid spectral basis dimension");
        }
        const T rank_tolerance = static_cast<T>(8.0*block_size_
            *static_cast<double>(std::numeric_limits<T>::epsilon()));
        if (mgsch_checked<T>(right_basis_, dimension(), block_size_,
                             rank_tolerance) == T(0)
            || mgsch_checked<T>(left_basis_, dimension(), block_size_,
                                rank_tolerance) == T(0)) {
            throw std::runtime_error(
                "linearly dependent vectors in spectral basis");
        }

        const int count = dimension();
        gram_.resize(static_cast<std::size_t>(count)*count);
        inverse_gram_.resize(static_cast<std::size_t>(count)*count);
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < count; ++j) {
                gram_[static_cast<std::size_t>(i)*count+j] = blas::dot(
                    block_size_, left_basis_[i].data(), 1,
                    right_basis_[j].data(), 1);
            }
        }

        min_pivot_ = inverse_general_matrix(
            inverse_gram_.data(), gram_.data(), count);
        const double inverse_norm = min_pivot_ > T(0)
            ? infinity_norm(inverse_gram_, count)
            : std::numeric_limits<double>::infinity();
        gram_condition_number_ = min_pivot_ > T(0)
            ? infinity_norm(gram_, count)*inverse_norm
            : std::numeric_limits<double>::infinity();
        projector_condition_number_ = std::max(
            gram_condition_number_, inverse_norm);
        if (!std::isfinite(projector_condition_number_)
            || projector_condition_number_ > condition_limit) {
            throw std::runtime_error(
                "ill-conditioned spectral basis in block (m="
                +std::to_string(m_)+",l="+std::to_string(l_)
                +"): condition="
                +std::to_string(projector_condition_number_)
                +", limit="+std::to_string(condition_limit));
        }
    }

    int m() const {
        return m_;
    }

    int l() const {
        return l_;
    }

    int block_size() const {
        return block_size_;
    }

    int dimension() const {
        return static_cast<int>(right_basis_.size());
    }

    int phase_count() const {
        return phase_count_;
    }

    int radial_size() const {
        return radial_size_;
    }

    bool pressure_gauge_fixed() const {
        return pressure_gauge_fixed_;
    }

    double condition_number() const {
        return projector_condition_number_;
    }

    double gram_condition_number() const {
        return gram_condition_number_;
    }

    double min_pivot() const {
        return static_cast<double>(min_pivot_);
    }

    const std::vector<std::vector<T>>& right_basis() const {
        return right_basis_;
    }

    const std::vector<std::vector<T>>& left_basis() const {
        return left_basis_;
    }

    const std::vector<T>& gram() const {
        return gram_;
    }

    const std::vector<T>& inverse_gram() const {
        return inverse_gram_;
    }

    // Coordinates in the orthonormalized real right basis. They are the
    // solution of G*a=Q_L^T*q and therefore remain meaningful for an oblique
    // projector. A complex eigenpair occupies two real coordinates.
    void coordinates(T* result, const T* state) const {
        const int count = dimension();
        std::vector<T> rhs(count);
        std::fill(result, result+count, T(0));
        for (int i = 0; i < count; ++i) {
            rhs[i] = blas::dot(block_size_, left_basis_[i].data(), 1,
                               state, 1);
        }
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < count; ++j) {
                result[i] += inverse_gram_[
                    static_cast<std::size_t>(i)*count+j]*rhs[j];
            }
        }
    }

    void project(T* result, const T* state) const {
        const int count = dimension();
        std::vector<T> coefficients(count);
        coordinates(coefficients.data(), state);

        std::fill(result, result+block_size_, T(0));
        for (int i = 0; i < count; ++i) {
            blas::axpy(block_size_, coefficients[i], right_basis_[i].data(),
                       1, result, 1);
        }
    }

    void remove(T* result, const T* state) const {
        std::vector<T> unstable(block_size_);
        project(unstable.data(), state);
        for (int i = 0; i < block_size_; ++i) {
            result[i] = state[i]-unstable[i];
        }
    }

private:
    int m_ = -1;
    int l_ = -1;
    int block_size_ = 0;
    int phase_count_ = 0;
    int radial_size_ = 0;
    bool pressure_gauge_fixed_ = false;
    T min_pivot_ = 0;
    double gram_condition_number_ = std::numeric_limits<double>::infinity();
    double projector_condition_number_ =
        std::numeric_limits<double>::infinity();
    std::vector<std::vector<T>> right_basis_;
    std::vector<std::vector<T>> left_basis_;
    std::vector<T> gram_;
    std::vector<T> inverse_gram_;

    void validate_metadata(const Mode& mode) const {
        if (!mode.filterable_unstable()) {
            throw std::invalid_argument(
                "spectral projector received an unaccepted mode");
        }
        if (mode.m != m_ || mode.l != l_ || mode.block_size != block_size_
            || mode.phase_count != phase_count_
            || mode.radial_size != radial_size_
            || mode.pressure_gauge_fixed != pressure_gauge_fixed_) {
            throw std::invalid_argument(
                "inconsistent spectral mode metadata within a block");
        }
        const std::size_t expected =
            static_cast<std::size_t>(mode.column_count)*block_size_;
        if (mode.column_count < 1 || mode.column_count > 2
            || mode.right_columns.size() != expected
            || mode.left_columns.size() != expected) {
            throw std::invalid_argument("invalid real spectral column layout");
        }
    }

    void append_columns(const std::vector<T>& columns, int count,
                        std::vector<std::vector<T>>& basis) const {
        for (int column = 0; column < count; ++column) {
            const auto begin = columns.begin()
                +static_cast<std::size_t>(column)*block_size_;
            basis.emplace_back(begin, begin+block_size_);
        }
    }

    static double infinity_norm(const std::vector<T>& matrix, int n) {
        double result = 0;
        for (int row = 0; row < n; ++row) {
            double sum = 0;
            for (int column = 0; column < n; ++column) {
                sum += std::abs(static_cast<double>(
                    matrix[static_cast<std::size_t>(row)*n+column]));
            }
            result = std::max(result, sum);
        }
        return result;
    }
};

template<typename T>
class NSCylSpectralProjector {
public:
    explicit NSCylSpectralProjector(const NSCylSpectralModeSet<T>& mode_set,
                                    double condition_limit) {
        if (!(condition_limit >= 1)) {
            throw std::invalid_argument(
                "spectral condition limit must be at least one");
        }
        std::map<std::pair<int, int>, std::vector<NSCylSpectralMode<T>>> groups;
        for (const auto& mode : mode_set.modes()) {
            groups[{mode.m, mode.l}].push_back(mode);
        }
        for (auto& [index, modes] : groups) {
            blocks_.emplace_back(modes, condition_limit);
        }
    }

    const std::vector<NSCylSpectralBlockProjector<T>>& blocks() const {
        return blocks_;
    }

    const NSCylSpectralBlockProjector<T>* find_block(int m, int l) const {
        const auto iterator = std::lower_bound(
            blocks_.begin(), blocks_.end(), std::pair<int, int>{m, l},
            [](const auto& block, const auto& index) {
                return std::pair<int, int>{block.m(), block.l()} < index;
            });
        if (iterator == blocks_.end() || iterator->m() != m
            || iterator->l() != l) {
            return nullptr;
        }
        return &*iterator;
    }

    int real_dimension() const {
        int result = 0;
        for (const auto& block : blocks_) {
            result += block.dimension();
        }
        return result;
    }

private:
    std::vector<NSCylSpectralBlockProjector<T>> blocks_;
};

} // namespace fdm
