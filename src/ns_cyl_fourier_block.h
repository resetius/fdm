#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fft.h"
#include "ns_cyl.h"
#include "ns_cyl_state.h"

namespace fdm {

// Two-dimensional periodic transform in the real packed convention used by
// FFT::pFFT_1/pFFT (Samarskii--Nikolaev): q stores cosine at q and sine at
// N-q. Analysis uses 2/N and synthesis uses 1, hence they are inverse.
template<typename T>
class PeriodicPackedFFT2 {
public:
    PeriodicPackedFFT2(int nphi, int nz)
        : nphi_(nphi)
        , nz_(nz)
#ifdef HAVE_FFTW3
        , fft_phi_(nphi)
        , fft_z_(nz)
#else
        , table_phi_(nphi)
        , table_z_(nz)
        , fft_phi_(table_phi_, nphi)
        , fft_z_(table_z_, nz)
#endif
        , temporary_(static_cast<std::size_t>(nphi)*nz)
        , line_in_(std::max(nphi, nz))
        , line_out_(std::max(nphi, nz))
    { }

    int size() const {
        return nphi_*nz_;
    }

    // packed Fourier coefficients -> values on the (phi,z) grid
    void synthesis(const T* coefficients, T* values) {
        for (int i = 0; i < nphi_; ++i) {
            for (int k = 0; k < nz_; ++k) {
                line_in_[k] = coefficients[index(i, k)];
            }
            fft_z_.pFFT(line_out_.data(), line_in_.data(), T(1));
            for (int k = 0; k < nz_; ++k) {
                temporary_[index(i, k)] = line_out_[k];
            }
        }

        for (int k = 0; k < nz_; ++k) {
            for (int i = 0; i < nphi_; ++i) {
                line_in_[i] = temporary_[index(i, k)];
            }
            fft_phi_.pFFT(line_out_.data(), line_in_.data(), T(1));
            for (int i = 0; i < nphi_; ++i) {
                values[index(i, k)] = line_out_[i];
            }
        }
    }

    // values on the (phi,z) grid -> packed Fourier coefficients
    void analysis(const T* values, T* coefficients) {
        const T phi_scale = T(2)/nphi_;
        const T z_scale = T(2)/nz_;

        for (int k = 0; k < nz_; ++k) {
            for (int i = 0; i < nphi_; ++i) {
                line_in_[i] = values[index(i, k)];
            }
            fft_phi_.pFFT_1(line_out_.data(), line_in_.data(), phi_scale);
            for (int i = 0; i < nphi_; ++i) {
                temporary_[index(i, k)] = line_out_[i];
            }
        }

        for (int i = 0; i < nphi_; ++i) {
            for (int k = 0; k < nz_; ++k) {
                line_in_[k] = temporary_[index(i, k)];
            }
            fft_z_.pFFT_1(line_out_.data(), line_in_.data(), z_scale);
            for (int k = 0; k < nz_; ++k) {
                coefficients[index(i, k)] = line_out_[k];
            }
        }
    }

private:
    int nphi_;
    int nz_;

#ifdef HAVE_FFTW3
    FFT_fftw3<T> fft_phi_;
    FFT_fftw3<T> fft_z_;
#else
    FFTTable<T> table_phi_;
    FFTTable<T> table_z_;
    FFT<T> fft_phi_;
    FFT<T> fft_z_;
#endif

    std::vector<T> temporary_;
    std::vector<T> line_in_;
    std::vector<T> line_out_;

    std::size_t index(int i, int k) const {
        return static_cast<std::size_t>(i)*nz_+k;
    }
};

// A correctness-first Fourier block of the existing real NSCyl::L_step.
// The block is expressed directly in the real tensor-product packing used by
// PeriodicPackedFFT2. For non-endpoint m and l it contains four phase sections
// (cos*cos, cos*sin, sin*cos, sin*sin), not complex-valued degrees of freedom.
template<typename T, bool check=false>
class NSCylFourierBlockReference {
public:
    using value_type = T;
    using Task = NSCyl<T, check, tensor_flag::periodic>;
    using tensor = typename Task::tensor;
    using StateLayout = NSCylStateLayout<T>;
    using Component = typename StateLayout::Component;

    NSCylFourierBlockReference(const Config& config, int m, int l,
                               int operator_steps=1)
        : ns_(config)
        , layout_(ns_)
        , fft_(ns_.nphi, ns_.nz)
        , m_(m)
        , l_(l)
        , phi_indices_(packed_indices(m, ns_.nphi))
        , z_indices_(packed_indices(l, ns_.nz))
        , radial_size_(layout_.radial_size)
        , full_block_size_(radial_size_*phase_count())
        , pressure_gauge_fixed_(m == 0 && l == 0)
        , block_size_(full_block_size_-(pressure_gauge_fixed_ ? 1 : 0))
        , operator_steps_(operator_steps)
        , coefficients_(fft_.size())
        , values_(fft_.size())
        , full_block_(full_block_size_)
    {
        if (m < 0 || m > ns_.nphi/2) {
            throw std::invalid_argument("azimuthal Fourier index is outside [0,nphi/2]");
        }
        if (l < 0 || l > ns_.nz/2) {
            throw std::invalid_argument("axial Fourier index is outside [0,nz/2]");
        }
        if (operator_steps <= 0) {
            throw std::invalid_argument("operator_steps must be positive");
        }
        initialize_couette_base();
    }

    int radial_size() const {
        return radial_size_;
    }

    int phase_count() const {
        return static_cast<int>(phi_indices_.size()*z_indices_.size());
    }

    int size() const {
        return block_size_;
    }

    int full_size() const {
        return full_block_size_;
    }

    bool pressure_gauge_fixed() const {
        return pressure_gauge_fixed_;
    }

    int operator_steps() const {
        return operator_steps_;
    }

    int m() const {
        return m_;
    }

    int l() const {
        return l_;
    }

    const std::vector<int>& phi_indices() const {
        return phi_indices_;
    }

    const std::vector<int>& z_indices() const {
        return z_indices_;
    }

    double last_fourier_leakage() const {
        return last_fourier_leakage_;
    }

    Task& task() {
        return ns_;
    }

    const StateLayout& state_layout() const {
        return layout_;
    }

    // Put a packed block vector into the full physical NSCyl state.
    void lift(const T* x) {
        std::fill(ns_.u.vec, ns_.u.vec+ns_.u.size, T(0));
        std::fill(ns_.v.vec, ns_.v.vec+ns_.v.size, T(0));
        std::fill(ns_.w.vec, ns_.w.vec+ns_.w.size, T(0));
        std::fill(ns_.p.vec, ns_.p.vec+ns_.p.size, T(0));

        const T* full_x = x;
        if (pressure_gauge_fixed_) {
            layout_.expand_zero_gauge_block(ns_, x, full_block_.data());
            full_x = full_block_.data();
        }
        layout_.for_each_radial([&](Component component, int j, int index) {
            lift_radial_slice(field(component), j, index, full_x);
        });
    }

    // Extract the selected packed block and report energy leaked to all other
    // Fourier slots. The leakage is diagnostic; it is not folded into y.
    void extract(T* y) {
        double other_norm2 = 0;
        T* full_y = pressure_gauge_fixed_ ? full_block_.data() : y;
        layout_.for_each_radial([&](Component component, int j, int index) {
            extract_radial_slice(field(component), j, index, full_y,
                                 other_norm2);
        });

        if (pressure_gauge_fixed_) {
            layout_.reduce_zero_gauge_block(ns_, full_y, y);
            layout_.expand_zero_gauge_block(ns_, y, full_block_.data());
            full_y = full_block_.data();
        }
        double selected_norm2 = 0;
        for (int index = 0; index < full_block_size_; ++index) {
            const double value = full_y[index];
            selected_norm2 += value*value;
        }
        const double total = selected_norm2+other_norm2;
        last_fourier_leakage_ = total > 0
            ? std::sqrt(other_norm2/total)
            : 0;
    }

    void apply(T* y, const T* x) {
        lift(x);
        for (int step = 0; step < operator_steps_; ++step) {
            ns_.L_step();
        }
        extract(y);
    }

private:
    Task ns_;
    StateLayout layout_;
    PeriodicPackedFFT2<T> fft_;
    int m_;
    int l_;
    std::vector<int> phi_indices_;
    std::vector<int> z_indices_;
    int radial_size_;
    int full_block_size_;
    bool pressure_gauge_fixed_;
    int block_size_;
    int operator_steps_;
    std::vector<T> coefficients_;
    std::vector<T> values_;
    std::vector<T> full_block_;
    double last_fourier_leakage_ = 0;

    static std::vector<int> packed_indices(int q, int n) {
        if (q < 0 || q > n/2) {
            throw std::invalid_argument("packed Fourier frequency is outside [0,N/2]");
        }
        if (q == 0 || 2*q == n) {
            return {q};
        }
        return {q, n-q};
    }

    void initialize_couette_base() {
        layout_.initialize_couette_linearization(ns_);

        // L_step advances a perturbation with homogeneous wall conditions;
        // the moving-wall velocity is already contained in w0.
        ns_.U0 = 0;
    }

    tensor& field(Component component) {
        switch (component) {
        case Component::u: return ns_.u;
        case Component::v: return ns_.v;
        case Component::w: return ns_.w;
        case Component::p: return ns_.p;
        }
        throw std::logic_error("unknown NSCyl state component");
    }

    std::size_t plane_index(int i, int k) const {
        return static_cast<std::size_t>(i)*ns_.nz+k;
    }

    bool selected_coefficient(int i, int k) const {
        return std::find(phi_indices_.begin(), phi_indices_.end(), i)
                   != phi_indices_.end()
            && std::find(z_indices_.begin(), z_indices_.end(), k)
                   != z_indices_.end();
    }

    int phase_index(int i, int k) const {
        const auto pi = std::find(phi_indices_.begin(), phi_indices_.end(), i);
        const auto zi = std::find(z_indices_.begin(), z_indices_.end(), k);
        if (pi == phi_indices_.end() || zi == z_indices_.end()) {
            throw std::logic_error("coefficient does not belong to this Fourier block");
        }
        return static_cast<int>((pi-phi_indices_.begin())*z_indices_.size()
                                +(zi-z_indices_.begin()));
    }

    void lift_radial_slice(tensor& field, int j, int state_index, const T* x) {
        std::fill(coefficients_.begin(), coefficients_.end(), T(0));
        for (int i : phi_indices_) {
            for (int k : z_indices_) {
                const int phase = phase_index(i, k);
                coefficients_[plane_index(i, k)] =
                    x[phase*radial_size_+state_index];
            }
        }

        fft_.synthesis(coefficients_.data(), values_.data());
        for (int i = 0; i < ns_.nphi; ++i) {
            for (int k = 0; k < ns_.nz; ++k) {
                field[i][k][j] = values_[plane_index(i, k)];
            }
        }
    }

    void extract_radial_slice(tensor& field, int j, int state_index, T* y,
                              double& other_norm2) {
        for (int i = 0; i < ns_.nphi; ++i) {
            for (int k = 0; k < ns_.nz; ++k) {
                values_[plane_index(i, k)] = field[i][k][j];
            }
        }
        fft_.analysis(values_.data(), coefficients_.data());

        for (int i = 0; i < ns_.nphi; ++i) {
            for (int k = 0; k < ns_.nz; ++k) {
                const T value = coefficients_[plane_index(i, k)];
                const double square = static_cast<double>(value)*value;
                if (selected_coefficient(i, k)) {
                    const int phase = phase_index(i, k);
                    y[phase*radial_size_+state_index] = value;
                } else {
                    other_norm2 += square;
                }
            }
        }
    }
};

} // namespace fdm
