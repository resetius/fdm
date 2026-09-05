#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

#include "ns_cyl_fourier_block.h"
#include "ns_cyl_sycl.h"

namespace fdm {

template<typename T>
class NSCylSyclFourierBlockReference {
public:
    using value_type = T;
    using Component = typename NSCylStateLayout<T>::Component;

    NSCylSyclFourierBlockReference(
        sycl::queue& queue, const Config& config, int m, int l,
        int operator_steps=1)
        : queue_(queue)
        , ns_(queue,
              config.get("ns", "nr", 32),
              config.get("ns", "nz", 32),
              config.get("ns", "nphi", 32),
              static_cast<T>(config.get("ns", "r", M_PI/2)),
              static_cast<T>(config.get("ns", "R", M_PI)),
              static_cast<T>(config.get("ns", "h2", 10.0)
                             -config.get("ns", "h1", 0.0)),
              T(0),
              static_cast<T>(config.get("ns", "Re", 1.0)),
              static_cast<T>(config.get("ns", "dt", 0.001)))
        , layout_(ns_.nr, ns_.nz, ns_.nphi)
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
            throw std::invalid_argument(
                "azimuthal Fourier index is outside [0,nphi/2]");
        }
        if (l < 0 || l > ns_.nz/2) {
            throw std::invalid_argument(
                "axial Fourier index is outside [0,nz/2]");
        }
        if (operator_steps <= 0) {
            throw std::invalid_argument("operator_steps must be positive");
        }
        ns_.initialize_couette_linearization(
            static_cast<T>(config.get("ns", "u0", 1.0)),
            static_cast<T>(config.get(
                "spectral", "base_outer_radius",
                config.get("ns", "R", M_PI))));
    }

    int radial_size() const { return radial_size_; }
    int phase_count() const {
        return static_cast<int>(phi_indices_.size()*z_indices_.size());
    }
    int size() const { return block_size_; }
    int full_size() const { return full_block_size_; }
    bool pressure_gauge_fixed() const { return pressure_gauge_fixed_; }
    int operator_steps() const { return operator_steps_; }
    int m() const { return m_; }
    int l() const { return l_; }
    double last_fourier_leakage() const { return last_fourier_leakage_; }
    NSCylSycl<T>& task() { return ns_; }

    void lift(const T* x) {
        clear_state();
        const T* full_x = x;
        if (pressure_gauge_fixed_) {
            layout_.expand_zero_gauge_block(ns_, x, full_block_.data());
            full_x = full_block_.data();
        }
        layout_.for_each_radial([&](Component component, int j, int index) {
            lift_radial_slice(field(component), j, index, full_x);
        });
    }

    void extract(T* y) {
        double other_norm2 = 0;
        T* full_y = pressure_gauge_fixed_ ? full_block_.data() : y;
        layout_.for_each_radial([&](Component component, int j, int index) {
            extract_radial_slice(
                field(component), j, index, full_y, other_norm2);
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
            ns_.L_step_fourier_block(m_, l_);
        }
        queue_.wait();
        extract(y);
    }

private:
    sycl::queue& queue_;
    NSCylSycl<T> ns_;
    NSCylStateLayout<T> layout_;
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
            throw std::invalid_argument(
                "packed Fourier frequency is outside [0,N/2]");
        }
        if (q == 0 || 2*q == n) {
            return {q};
        }
        return {q, n-q};
    }

    void clear_state() {
        auto u = ns_.ua();
        auto v = ns_.va();
        auto w = ns_.wa();
        auto p = ns_.pa();
        for (int i = 0; i < ns_.nphi; ++i) {
            for (int k = 0; k < ns_.nz; ++k) {
                for (int j = -1; j <= ns_.nr+1; ++j) {
                    u(i, k, j) = T(0);
                }
                for (int j = 0; j <= ns_.nr+1; ++j) {
                    v(i, k, j) = T(0);
                    w(i, k, j) = T(0);
                    p(i, k, j) = T(0);
                }
            }
        }
    }

    CylAcc<T> field(Component component) const {
        switch (component) {
        case Component::u: return ns_.ua();
        case Component::v: return ns_.va();
        case Component::w: return ns_.wa();
        case Component::p: return ns_.pa();
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
            throw std::logic_error(
                "coefficient does not belong to this Fourier block");
        }
        return static_cast<int>((pi-phi_indices_.begin())*z_indices_.size()
                                +(zi-z_indices_.begin()));
    }

    void lift_radial_slice(
        CylAcc<T> field, int j, int state_index, const T* x)
    {
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
                field(i, k, j) = values_[plane_index(i, k)];
            }
        }
    }

    void extract_radial_slice(
        CylAcc<T> field, int j, int state_index, T* y,
        double& other_norm2)
    {
        for (int i = 0; i < ns_.nphi; ++i) {
            for (int k = 0; k < ns_.nz; ++k) {
                values_[plane_index(i, k)] = field(i, k, j);
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
