#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "blas.h"

namespace fdm {

// LAPACK stores a real eigenvector in one column and a complex conjugate pair
// in two adjacent real columns. The state vectors below retain that layout.
template<typename T>
struct NSCylSpectralMode {
    int m = -1;
    int l = -1;
    int phase_count = 0;
    int radial_size = 0;
    int block_size = 0;
    bool pressure_gauge_fixed = false;

    std::complex<T> multiplier{};
    double growth_rate = -std::numeric_limits<double>::infinity();
    double frequency = 0;
    double right_residual = std::numeric_limits<double>::infinity();
    double left_residual = std::numeric_limits<double>::infinity();
    double block_condition_number =
        std::numeric_limits<double>::quiet_NaN();
    bool growing = false;
    bool residual_accepted = false;

    int column_count = 0;
    std::vector<T> right_columns;
    std::vector<T> left_columns;

    bool filterable_unstable() const {
        return growing && residual_accepted;
    }
};

template<typename T>
struct NSCylDenseBlockSpectrum {
    int m = -1;
    int l = -1;
    int phase_count = 0;
    int radial_size = 0;
    int block_size = 0;
    int operator_steps = 0;
    int operator_calls = 0;
    bool pressure_gauge_fixed = false;
    double duration = 0;
    double growth_tolerance = 0;
    double residual_tolerance = 0;
    double max_fourier_leakage = 0;
    double max_right_residual = 0;
    double max_left_residual = 0;

    // These arrays retain the one-entry-per-LAPACK-column representation.
    std::vector<std::complex<T>> eigenvalues;
    std::vector<double> right_residuals;
    std::vector<double> left_residuals;
    std::vector<NSCylSpectralMode<T>> modes;

    int growing_dimension() const {
        int result = 0;
        for (const auto& mode : modes) {
            if (mode.growing) {
                result += mode.column_count;
            }
        }
        return result;
    }

    int filterable_unstable_dimension() const {
        int result = 0;
        for (const auto& mode : modes) {
            if (mode.filterable_unstable()) {
                result += mode.column_count;
            }
        }
        return result;
    }

    int filterable_unstable_group_count() const {
        return static_cast<int>(std::count_if(
            modes.begin(), modes.end(), [](const auto& mode) {
                return mode.filterable_unstable();
            }));
    }
};

template<typename T>
class NSCylSpectralModeSet {
public:
    void append_filterable_mode(NSCylSpectralMode<T> mode) {
        if (!mode.filterable_unstable()) {
            throw std::invalid_argument(
                "NSCylSpectralModeSet accepts only filterable unstable modes");
        }
        modes_.push_back(std::move(mode));
    }

    void append_filterable(const NSCylDenseBlockSpectrum<T>& spectrum) {
        for (const auto& mode : spectrum.modes) {
            if (mode.filterable_unstable()) {
                append_filterable_mode(mode);
            }
        }
    }

    void sort_by_block_and_growth() {
        std::stable_sort(modes_.begin(), modes_.end(),
            [](const auto& a, const auto& b) {
                if (a.m != b.m) {
                    return a.m < b.m;
                }
                if (a.l != b.l) {
                    return a.l < b.l;
                }
                return a.growth_rate > b.growth_rate;
            });
    }

    int real_dimension() const {
        int result = 0;
        for (const auto& mode : modes_) {
            result += mode.column_count;
        }
        return result;
    }

    const std::vector<NSCylSpectralMode<T>>& modes() const {
        return modes_;
    }

    bool empty() const {
        return modes_.empty();
    }

    std::size_t size() const {
        return modes_.size();
    }

private:
    std::vector<NSCylSpectralMode<T>> modes_;
};

namespace ns_cyl_spectral_detail {

template<typename T>
double squared_norm(int n, const T* x) {
    double result = 0;
    for (int i = 0; i < n; ++i) {
        const double value = static_cast<double>(x[i]);
        result += value*value;
    }
    return result;
}

template<typename T>
void matvec(int n, const T* matrix, const T* x, T* y) {
    std::fill(y, y+n, T(0));
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < n; ++row) {
            y[row] += matrix[static_cast<std::size_t>(column)*n+row]
                *x[column];
        }
    }
}

template<typename T>
void transposed_matvec(int n, const T* matrix, const T* x, T* y) {
    for (int row = 0; row < n; ++row) {
        T result = 0;
        for (int column = 0; column < n; ++column) {
            result += matrix[static_cast<std::size_t>(row)*n+column]
                *x[column];
        }
        y[row] = result;
    }
}

inline double scaled_residual(double residual_squared, double vector_norm_squared,
                              double matrix_norm) {
    const double denominator = matrix_norm*std::sqrt(vector_norm_squared);
    if (denominator == 0) {
        return residual_squared == 0
            ? 0
            : std::numeric_limits<double>::infinity();
    }
    return std::sqrt(residual_squared)/denominator;
}

template<typename T>
void compute_residuals(int n, const T* matrix, const T* real,
                       const T* imaginary, const T* left, const T* right,
                       double matrix_norm, std::vector<double>& right_result,
                       std::vector<double>& left_result) {
    right_result.assign(n, 0);
    left_result.assign(n, 0);
    std::vector<T> real_image(n);
    std::vector<T> imaginary_image(n);

    for (int i = 0; i < n; ) {
        const int column_count = imaginary[i] == T(0) ? 1 : 2;
        if (i+column_count > n) {
            throw std::runtime_error("incomplete LAPACK conjugate pair");
        }

        const T wr = real[i];
        const T wi = imaginary[i];
        const T* vr = right+static_cast<std::size_t>(i)*n;
        const T* vl = left+static_cast<std::size_t>(i)*n;
        double residual_squared = 0;
        double vector_norm_squared = 0;

        if (column_count == 1) {
            matvec(n, matrix, vr, real_image.data());
            for (int row = 0; row < n; ++row) {
                real_image[row] -= wr*vr[row];
            }
            residual_squared = squared_norm(n, real_image.data());
            vector_norm_squared = squared_norm(n, vr);
        } else {
            const T* vi = vr+n;
            matvec(n, matrix, vr, real_image.data());
            matvec(n, matrix, vi, imaginary_image.data());
            for (int row = 0; row < n; ++row) {
                real_image[row] -= wr*vr[row]-wi*vi[row];
                imaginary_image[row] -= wr*vi[row]+wi*vr[row];
            }
            residual_squared = squared_norm(n, real_image.data())
                +squared_norm(n, imaginary_image.data());
            vector_norm_squared = squared_norm(n, vr)+squared_norm(n, vi);
        }
        const double right_residual = scaled_residual(
            residual_squared, vector_norm_squared, matrix_norm);
        std::fill(right_result.begin()+i,
                  right_result.begin()+i+column_count, right_residual);

        if (column_count == 1) {
            transposed_matvec(n, matrix, vl, real_image.data());
            for (int row = 0; row < n; ++row) {
                real_image[row] -= wr*vl[row];
            }
            residual_squared = squared_norm(n, real_image.data());
            vector_norm_squared = squared_norm(n, vl);
        } else {
            const T* li = vl+n;
            transposed_matvec(n, matrix, vl, real_image.data());
            transposed_matvec(n, matrix, li, imaginary_image.data());
            for (int row = 0; row < n; ++row) {
                real_image[row] -= wr*vl[row]+wi*li[row];
                imaginary_image[row] -= wr*li[row]-wi*vl[row];
            }
            residual_squared = squared_norm(n, real_image.data())
                +squared_norm(n, imaginary_image.data());
            vector_norm_squared = squared_norm(n, vl)+squared_norm(n, li);
        }
        const double left_residual = scaled_residual(
            residual_squared, vector_norm_squared, matrix_norm);
        std::fill(left_result.begin()+i,
                  left_result.begin()+i+column_count, left_residual);
        i += column_count;
    }
}

} // namespace ns_cyl_spectral_detail

// Analyze a real column-major matrix. A complex mode is represented by its
// positive-imaginary eigenvalue and the adjacent Re/Im LAPACK columns.
template<typename T>
NSCylDenseBlockSpectrum<T> analyze_ns_cyl_dense_matrix(
    const T* matrix, int n, double duration, double growth_tolerance,
    double residual_tolerance) {
    if (n <= 0) {
        throw std::invalid_argument("dense spectral matrix must be nonempty");
    }
    if (!(duration > 0)) {
        throw std::invalid_argument("spectral operator duration must be positive");
    }
    if (residual_tolerance < 0) {
        throw std::invalid_argument("spectral residual tolerance must be nonnegative");
    }

    NSCylDenseBlockSpectrum<T> result;
    result.block_size = n;
    result.duration = duration;
    result.growth_tolerance = growth_tolerance;
    result.residual_tolerance = residual_tolerance;

    std::vector<T> factored(matrix, matrix+static_cast<std::size_t>(n)*n);
    std::vector<T> real(n);
    std::vector<T> imaginary(n);
    std::vector<T> left(static_cast<std::size_t>(n)*n);
    std::vector<T> right(static_cast<std::size_t>(n)*n);
    std::vector<T> work(8*n);
    int info = 0;
    lapack::geev("V", "V", n, factored.data(), n, real.data(),
                 imaginary.data(), left.data(), n, right.data(), n,
                 work.data(), static_cast<int>(work.size()), &info);
    if (info != 0) {
        throw std::runtime_error("geev failed with info="+std::to_string(info));
    }

    const double matrix_norm = std::sqrt(
        ns_cyl_spectral_detail::squared_norm(
            static_cast<int>(static_cast<std::size_t>(n)*n), matrix));
    ns_cyl_spectral_detail::compute_residuals(
        n, matrix, real.data(), imaginary.data(), left.data(), right.data(),
        matrix_norm, result.right_residuals, result.left_residuals);

    result.eigenvalues.reserve(n);
    for (int i = 0; i < n; ++i) {
        result.eigenvalues.emplace_back(real[i], imaginary[i]);
        result.max_right_residual = std::max(
            result.max_right_residual, result.right_residuals[i]);
        result.max_left_residual = std::max(
            result.max_left_residual, result.left_residuals[i]);
    }

    for (int i = 0; i < n; ) {
        const int column_count = imaginary[i] == T(0) ? 1 : 2;
        if (i+column_count > n) {
            throw std::runtime_error("incomplete LAPACK conjugate pair");
        }
        if (column_count == 2 && imaginary[i] < T(0)) {
            throw std::runtime_error("unexpected LAPACK conjugate-pair order");
        }

        NSCylSpectralMode<T> mode;
        mode.block_size = n;
        mode.multiplier = {real[i], imaginary[i]};
        const double magnitude = std::abs(mode.multiplier);
        mode.growth_rate = magnitude > 0
            ? std::log(magnitude)/duration
            : -std::numeric_limits<double>::infinity();
        mode.frequency = std::atan2(
            static_cast<double>(imaginary[i]),
            static_cast<double>(real[i]))/duration;
        mode.right_residual = result.right_residuals[i];
        mode.left_residual = result.left_residuals[i];
        mode.growing = mode.growth_rate > growth_tolerance;
        mode.residual_accepted =
            mode.right_residual <= residual_tolerance
            && mode.left_residual <= residual_tolerance;
        mode.column_count = column_count;
        mode.right_columns.assign(
            right.begin()+static_cast<std::size_t>(i)*n,
            right.begin()+static_cast<std::size_t>(i+column_count)*n);
        mode.left_columns.assign(
            left.begin()+static_cast<std::size_t>(i)*n,
            left.begin()+static_cast<std::size_t>(i+column_count)*n);
        result.modes.push_back(std::move(mode));
        i += column_count;
    }
    return result;
}

template<typename Block>
NSCylDenseBlockSpectrum<typename Block::value_type>
solve_ns_cyl_dense_block(Block& block, double dt, double growth_tolerance,
                         double residual_tolerance) {
    using T = typename Block::value_type;
    const int n = block.size();
    std::vector<T> matrix(static_cast<std::size_t>(n)*n);
    std::vector<T> basis(n, T(0));
    std::vector<T> image(n);
    double max_leakage = 0;
    for (int column = 0; column < n; ++column) {
        basis[column] = T(1);
        block.apply(image.data(), basis.data());
        basis[column] = T(0);
        max_leakage = std::max(max_leakage, block.last_fourier_leakage());
        std::copy(image.begin(), image.end(),
                  matrix.begin()+static_cast<std::size_t>(column)*n);
    }

    auto result = analyze_ns_cyl_dense_matrix(
        matrix.data(), n, block.operator_steps()*dt,
        growth_tolerance, residual_tolerance);
    result.m = block.m();
    result.l = block.l();
    result.phase_count = block.phase_count();
    result.radial_size = block.radial_size();
    result.operator_steps = block.operator_steps();
    result.operator_calls = n;
    result.pressure_gauge_fixed = block.pressure_gauge_fixed();
    result.max_fourier_leakage = max_leakage;
    for (auto& mode : result.modes) {
        mode.m = result.m;
        mode.l = result.l;
        mode.phase_count = result.phase_count;
        mode.radial_size = result.radial_size;
        mode.pressure_gauge_fixed = result.pressure_gauge_fixed;
    }
    return result;
}

} // namespace fdm
