#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ns_cyl_fourier_block.h"
#include "ns_cyl_spectral_projector.h"
#include "ns_cyl_state.h"

namespace fdm {

enum class NSCylSpectralRemoval {
    unstable_eigenspace,
    whole_fourier_blocks
};

struct NSCylSpectralBlockFilterDiagnostics {
    int m = -1;
    int l = -1;
    double block_norm = 0;
    double removed_norm = 0;
    double remaining_unstable_norm = 0;
};

struct NSCylSpectralFilterDiagnostics {
    double packed_perturbation_norm = 0;
    double removed_norm = 0;
    double remaining_unstable_norm = 0;
    std::vector<NSCylSpectralBlockFilterDiagnostics> blocks;
};

// Applies real block projectors to a complete physical NSCyl state. The
// transform and all stored coefficients retain the Samarskii--Nikolaev real
// packing, so no complex full-field representation is introduced.
template<typename T>
class NSCylSpectralFilter {
public:
    NSCylSpectralFilter(int nr, int nphi, int nz,
                        NSCylSpectralProjector<T> projector)
        : layout_(nr, nz, nphi)
        , fft_(nphi, nz)
        , projector_(std::move(projector))
        , physical_(layout_.state_size)
        , packed_fourier_(layout_.state_size)
        , values_(fft_.size())
        , coefficients_(fft_.size()) {
        validate_blocks();
    }

    const NSCylSpectralProjector<T>& projector() const {
        return projector_;
    }

    template<typename Task>
    NSCylSpectralFilterDiagnostics measure(
        Task& state, const std::vector<T>& reference,
        NSCylSpectralRemoval removal =
            NSCylSpectralRemoval::unstable_eigenspace) {
        return execute(state, reference, removal, false);
    }

    template<typename Task>
    NSCylSpectralFilterDiagnostics remove(
        Task& state, const std::vector<T>& reference,
        NSCylSpectralRemoval removal =
            NSCylSpectralRemoval::unstable_eigenspace) {
        return execute(state, reference, removal, true);
    }

private:
    using Layout = NSCylStateLayout<T>;
    using Component = typename Layout::Component;
    using BlockProjector = NSCylSpectralBlockProjector<T>;

    Layout layout_;
    PeriodicPackedFFT2<T> fft_;
    NSCylSpectralProjector<T> projector_;
    std::vector<T> physical_;
    std::vector<T> packed_fourier_;
    std::vector<T> values_;
    std::vector<T> coefficients_;
    std::vector<T> full_block_;
    std::vector<T> block_;
    std::vector<T> filtered_block_;
    std::vector<T> removed_block_;
    std::vector<T> remaining_unstable_;

    static std::vector<int> packed_indices(int frequency, int size) {
        if (frequency == 0 || 2*frequency == size) {
            return {frequency};
        }
        return {frequency, size-frequency};
    }

    int component_size(Component component) const {
        return component == Component::u ? layout_.nr-1 : layout_.nr;
    }

    int component_offset(Component component) const {
        switch (component) {
        case Component::u: return layout_.u_offset;
        case Component::v: return layout_.v_offset;
        case Component::w: return layout_.w_offset;
        case Component::p: return layout_.p_offset;
        }
        throw std::logic_error("unknown NSCyl state component");
    }

    int state_index(Component component, int i, int k, int j) const {
        return component_offset(component)
            +(i*layout_.nz+k)*component_size(component)+(j-1);
    }

    std::size_t plane_index(int i, int k) const {
        return static_cast<std::size_t>(i)*layout_.nz+k;
    }

    void validate_blocks() const {
        for (const auto& block : projector_.blocks()) {
            if (block.m() < 0 || block.m() > layout_.nphi/2
                || block.l() < 0 || block.l() > layout_.nz/2) {
                throw std::invalid_argument(
                    "spectral projector block is outside the filter grid");
            }
            const int phi_phases =
                (block.m() == 0 || 2*block.m() == layout_.nphi) ? 1 : 2;
            const int z_phases =
                (block.l() == 0 || 2*block.l() == layout_.nz) ? 1 : 2;
            const bool gauge_fixed = block.m() == 0 && block.l() == 0;
            const int expected_size = layout_.radial_size*phi_phases*z_phases
                -(gauge_fixed ? 1 : 0);
            if (block.phase_count() != phi_phases*z_phases
                || block.radial_size() != layout_.radial_size
                || block.block_size() != expected_size
                || block.pressure_gauge_fixed() != gauge_fixed) {
                throw std::invalid_argument(
                    "spectral projector layout does not match filter grid");
            }
        }
    }

    template<typename Function>
    void for_each_radial_slice(Function&& function) {
        layout_.for_each_radial(
            [&](Component component, int j, int radial_index) {
                function(component, j, radial_index);
            });
    }

    void analysis() {
        for_each_radial_slice(
            [&](Component component, int j, int) {
                for (int i = 0; i < layout_.nphi; ++i) {
                    for (int k = 0; k < layout_.nz; ++k) {
                        values_[plane_index(i, k)] =
                            physical_[state_index(component, i, k, j)];
                    }
                }
                fft_.analysis(values_.data(), coefficients_.data());
                for (int i = 0; i < layout_.nphi; ++i) {
                    for (int k = 0; k < layout_.nz; ++k) {
                        packed_fourier_[state_index(component, i, k, j)] =
                            coefficients_[plane_index(i, k)];
                    }
                }
            });
    }

    void synthesis() {
        for_each_radial_slice(
            [&](Component component, int j, int) {
                for (int i = 0; i < layout_.nphi; ++i) {
                    for (int k = 0; k < layout_.nz; ++k) {
                        coefficients_[plane_index(i, k)] =
                            packed_fourier_[state_index(component, i, k, j)];
                    }
                }
                fft_.synthesis(coefficients_.data(), values_.data());
                for (int i = 0; i < layout_.nphi; ++i) {
                    for (int k = 0; k < layout_.nz; ++k) {
                        physical_[state_index(component, i, k, j)] =
                            values_[plane_index(i, k)];
                    }
                }
            });
    }

    template<typename Geometry>
    void canonicalize_pressure_gauge(const Geometry& geometry) {
        long double weighted_sum = 0;
        long double weight = 0;
        for (int j = 1; j <= layout_.nr; ++j) {
            const long double radius = geometry.r0+(j-0.5L)*geometry.dr;
            weighted_sum += radius*static_cast<long double>(packed_fourier_[
                state_index(Component::p, 0, 0, j)]);
            weight += radius;
        }
        const T mean = static_cast<T>(weighted_sum/weight);
        for (int j = 1; j <= layout_.nr; ++j) {
            packed_fourier_[state_index(Component::p, 0, 0, j)] -= mean;
        }
    }

    template<typename Geometry>
    void gather_block(const BlockProjector& projector,
                      const Geometry& geometry) {
        const auto phi_indices = packed_indices(projector.m(), layout_.nphi);
        const auto z_indices = packed_indices(projector.l(), layout_.nz);
        full_block_.assign(
            static_cast<std::size_t>(projector.phase_count())
                *layout_.radial_size,
            T(0));

        int phase = 0;
        for (int i : phi_indices) {
            for (int k : z_indices) {
                for_each_radial_slice(
                    [&](Component component, int j, int radial_index) {
                        full_block_[static_cast<std::size_t>(phase)
                                        *layout_.radial_size+radial_index] =
                            packed_fourier_[state_index(component, i, k, j)];
                    });
                ++phase;
            }
        }

        block_.resize(projector.block_size());
        if (projector.pressure_gauge_fixed()) {
            layout_.reduce_zero_gauge_block(
                geometry, full_block_.data(), block_.data());
        } else {
            std::copy(full_block_.begin(), full_block_.end(), block_.begin());
        }
    }

    template<typename Geometry>
    void scatter_block(const BlockProjector& projector,
                       const Geometry& geometry, const T* block) {
        const T* full = block;
        if (projector.pressure_gauge_fixed()) {
            layout_.expand_zero_gauge_block(
                geometry, block, full_block_.data());
            full = full_block_.data();
        }

        const auto phi_indices = packed_indices(projector.m(), layout_.nphi);
        const auto z_indices = packed_indices(projector.l(), layout_.nz);
        int phase = 0;
        for (int i : phi_indices) {
            for (int k : z_indices) {
                for_each_radial_slice(
                    [&](Component component, int j, int radial_index) {
                        packed_fourier_[state_index(component, i, k, j)] =
                            full[static_cast<std::size_t>(phase)
                                     *layout_.radial_size+radial_index];
                    });
                ++phase;
            }
        }
    }

    static double norm(const std::vector<T>& values) {
        long double result = 0;
        for (T value : values) {
            const long double x = static_cast<long double>(value);
            result += x*x;
        }
        return std::sqrt(static_cast<double>(result));
    }

    template<typename Task>
    NSCylSpectralFilterDiagnostics execute(
        Task& state, const std::vector<T>& reference,
        NSCylSpectralRemoval removal, bool update_state) {
        if (state.nr != layout_.nr || state.nphi != layout_.nphi
            || state.nz != layout_.nz) {
            throw std::invalid_argument(
                "NSCyl state dimensions do not match spectral filter");
        }

        NSCylSpectralFilterDiagnostics result;
        if (projector_.blocks().empty()) {
            return result;
        }

        layout_.pack_difference(state, reference, physical_.data());
        analysis();
        canonicalize_pressure_gauge(state);
        result.packed_perturbation_norm = norm(packed_fourier_);

        long double removed_squared = 0;
        long double remaining_squared = 0;
        for (const auto& projector : projector_.blocks()) {
            gather_block(projector, state);
            filtered_block_.resize(projector.block_size());
            removed_block_.resize(projector.block_size());
            remaining_unstable_.resize(projector.block_size());

            if (removal == NSCylSpectralRemoval::whole_fourier_blocks) {
                removed_block_ = block_;
                std::fill(filtered_block_.begin(), filtered_block_.end(), T(0));
            } else {
                projector.project(removed_block_.data(), block_.data());
                for (int i = 0; i < projector.block_size(); ++i) {
                    filtered_block_[i] = block_[i]-removed_block_[i];
                }
            }
            projector.project(
                remaining_unstable_.data(), filtered_block_.data());

            NSCylSpectralBlockFilterDiagnostics block_result;
            block_result.m = projector.m();
            block_result.l = projector.l();
            block_result.block_norm = norm(block_);
            block_result.removed_norm = norm(removed_block_);
            block_result.remaining_unstable_norm = norm(remaining_unstable_);
            result.blocks.push_back(block_result);
            removed_squared += block_result.removed_norm
                *block_result.removed_norm;
            remaining_squared += block_result.remaining_unstable_norm
                *block_result.remaining_unstable_norm;

            if (update_state) {
                scatter_block(projector, state, filtered_block_.data());
            }
        }
        result.removed_norm = std::sqrt(static_cast<double>(removed_squared));
        result.remaining_unstable_norm =
            std::sqrt(static_cast<double>(remaining_squared));

        if (update_state) {
            synthesis();
            layout_.unpack_sum(state, reference, physical_.data());
        }
        return result;
    }
};

} // namespace fdm
