#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "ns_cyl_fourier_block.h"
#include "ns_cyl_state.h"

namespace fdm {

// Parseval decomposition of the cylindrical velocity energy by real packed
// (m,l) Fourier block. The basis weights are measured through synthesis, so
// endpoint cosine modes and interior sine/cosine modes use exactly the same
// normalization convention as PeriodicPackedFFT2.
template<typename T>
class NSCylFourierVelocityEnergy {
public:
    using Layout = NSCylStateLayout<T>;
    using Component = typename Layout::Component;

    NSCylFourierVelocityEnergy(int nr, int nz, int nphi)
        : layout_(nr, nz, nphi)
        , fft_(nphi, nz)
        , values_(fft_.size())
        , coefficients_(fft_.size())
        , basis_norm2_(fft_.size())
    {
        build_basis_norms();
    }

    int m_count() const { return layout_.nphi/2+1; }
    int l_count() const { return layout_.nz/2+1; }

    std::size_t block_index(int m, int l) const {
        if (m < 0 || m >= m_count() || l < 0 || l >= l_count()) {
            throw std::out_of_range("Fourier velocity-energy block index");
        }
        return static_cast<std::size_t>(m)*l_count()+l;
    }

    template<typename Geometry>
    std::vector<double> energies(const Geometry& geometry,
                                 const std::vector<T>& state) {
        if (static_cast<int>(state.size()) != layout_.state_size) {
            throw std::invalid_argument(
                "Fourier velocity energy state has the wrong size");
        }
        std::vector<double> result(
            static_cast<std::size_t>(m_count())*l_count(), 0.0);
        const long double cell_measure = static_cast<long double>(geometry.dr)
            *geometry.dphi*geometry.dz;
        layout_.for_each_radial([&](Component component, int j, int) {
            if (component == Component::p) {
                return;
            }
            for (int i = 0; i < layout_.nphi; ++i) {
                for (int k = 0; k < layout_.nz; ++k) {
                    values_[plane_index(i, k)] = state[
                        state_index(component, i, k, j)];
                }
            }
            fft_.analysis(values_.data(), coefficients_.data());
            const long double radius = component == Component::u
                ? geometry.r0+j*geometry.dr
                : geometry.r0+(j-0.5L)*geometry.dr;
            for (int i = 0; i < layout_.nphi; ++i) {
                const int m = std::min(i, layout_.nphi-i);
                for (int k = 0; k < layout_.nz; ++k) {
                    const int l = std::min(k, layout_.nz-k);
                    const std::size_t index = plane_index(i, k);
                    const long double value = coefficients_[index];
                    result[block_index(m, l)] += static_cast<double>(
                        cell_measure*radius*basis_norm2_[index]*value*value);
                }
            }
        });
        return result;
    }

private:
    Layout layout_;
    PeriodicPackedFFT2<T> fft_;
    std::vector<T> values_;
    std::vector<T> coefficients_;
    std::vector<long double> basis_norm2_;

    std::size_t plane_index(int i, int k) const {
        return static_cast<std::size_t>(i)*layout_.nz+k;
    }

    int state_index(Component component, int i, int k, int j) const {
        int offset = 0;
        int radial_size = layout_.nr;
        switch (component) {
        case Component::u:
            offset = layout_.u_offset;
            radial_size = layout_.nr-1;
            break;
        case Component::v: offset = layout_.v_offset; break;
        case Component::w: offset = layout_.w_offset; break;
        case Component::p: offset = layout_.p_offset; break;
        }
        return offset+(i*layout_.nz+k)*radial_size+j-1;
    }

    void build_basis_norms() {
        std::fill(coefficients_.begin(), coefficients_.end(), T(0));
        for (std::size_t index = 0; index < coefficients_.size(); ++index) {
            coefficients_[index] = T(1);
            fft_.synthesis(coefficients_.data(), values_.data());
            long double norm2 = 0;
            for (T value : values_) {
                norm2 += static_cast<long double>(value)*value;
            }
            basis_norm2_[index] = norm2;
            coefficients_[index] = T(0);
        }
    }
};

} // namespace fdm
