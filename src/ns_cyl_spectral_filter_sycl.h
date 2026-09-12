#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fft/sycl/rfft_registers.h"
#include "ns_cyl_spectral_filter.h"

namespace fdm {

// Device implementation of the real-packed Fourier filter.  The nonlinear
// state, both periodic transforms, and the block projectors stay in shared
// USM.  A call waits only before returning host diagnostics.
template<typename T>
class NSCylSpectralFilterSycl {
    using Layout = NSCylStateLayout<T>;
    using Component = typename Layout::Component;
    using BlockProjector = NSCylSpectralBlockProjector<T>;

    struct KernelBlockInfo {
        int m;
        int l;
        int block_size;
        int dimension;
        int index_offset;
        int value_offset;
        int basis_offset;
        int inverse_offset;
        int coordinate_offset;
        int pressure_gauge_fixed;
    };

public:
    NSCylSpectralFilterSycl(sycl::queue& queue, int nr, int nphi, int nz,
                            NSCylSpectralProjector<T> projector)
        : queue_(require_in_order(queue))
        , layout_(nr, nz, nphi)
        , projector_(std::move(projector)) {
        build_projector_data();
        if (blocks_.empty()) {
            return;
        }
        if (!supported_fft_size(nphi) || !supported_fft_size(nz)) {
            throw std::invalid_argument(
                "SYCL spectral filter needs power-of-two periodic sizes "
                "between 4 and 256");
        }

        try {
            physical_ = allocate<T>(layout_.state_size);
            original_physical_ = allocate<T>(layout_.state_size);
            temporary_ = allocate<T>(layout_.state_size);
            packed_fourier_ = allocate<T>(layout_.state_size);
            original_fourier_ = allocate<T>(layout_.state_size);
            reference_ = allocate<T>(layout_.state_size);
            block_info_ = allocate<KernelBlockInfo>(blocks_.size());
            packed_indices_ = allocate<int>(packed_indices_host_.size());
            right_basis_ = allocate<T>(right_basis_host_.size());
            left_basis_ = allocate<T>(left_basis_host_.size());
            inverse_gram_ = allocate<T>(inverse_gram_host_.size());
            block_values_ = allocate<T>(total_block_values_);
            removed_values_ = allocate<T>(total_block_values_);
            remaining_values_ = allocate<T>(total_block_values_);
            coordinate_rhs_ = allocate<T>(total_coordinates_);
            coordinates_before_ = allocate<T>(total_coordinates_);
            coordinates_after_ = allocate<T>(total_coordinates_);

            const auto phi_twiddles =
                fft_sycl::make_twiddles<T>(layout_.nphi);
            const auto z_twiddles =
                fft_sycl::make_twiddles<T>(layout_.nz);
            twiddles_phi_ = allocate<T>(phi_twiddles.size());
            twiddles_z_ = allocate<T>(z_twiddles.size());

            std::copy(blocks_.begin(), blocks_.end(), block_info_);
            std::copy(packed_indices_host_.begin(), packed_indices_host_.end(),
                      packed_indices_);
            std::copy(right_basis_host_.begin(), right_basis_host_.end(),
                      right_basis_);
            std::copy(left_basis_host_.begin(), left_basis_host_.end(),
                      left_basis_);
            std::copy(inverse_gram_host_.begin(), inverse_gram_host_.end(),
                      inverse_gram_);
            std::copy(phi_twiddles.begin(), phi_twiddles.end(),
                      twiddles_phi_);
            std::copy(z_twiddles.begin(), z_twiddles.end(), twiddles_z_);
        } catch (...) {
            release();
            throw;
        }
    }

    NSCylSpectralFilterSycl(const NSCylSpectralFilterSycl&) = delete;
    NSCylSpectralFilterSycl& operator=(
        const NSCylSpectralFilterSycl&) = delete;

    ~NSCylSpectralFilterSycl() {
        queue_.wait();
        release();
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
    sycl::queue& queue_;
    Layout layout_;
    NSCylSpectralProjector<T> projector_;

    std::vector<KernelBlockInfo> blocks_;
    std::vector<int> packed_indices_host_;
    std::vector<T> right_basis_host_;
    std::vector<T> left_basis_host_;
    std::vector<T> inverse_gram_host_;
    int total_block_values_ = 0;
    int total_coordinates_ = 0;

    T* physical_ = nullptr;
    T* original_physical_ = nullptr;
    T* temporary_ = nullptr;
    T* packed_fourier_ = nullptr;
    T* original_fourier_ = nullptr;
    T* reference_ = nullptr;
    KernelBlockInfo* block_info_ = nullptr;
    int* packed_indices_ = nullptr;
    T* right_basis_ = nullptr;
    T* left_basis_ = nullptr;
    T* inverse_gram_ = nullptr;
    T* block_values_ = nullptr;
    T* removed_values_ = nullptr;
    T* remaining_values_ = nullptr;
    T* coordinate_rhs_ = nullptr;
    T* coordinates_before_ = nullptr;
    T* coordinates_after_ = nullptr;
    T* twiddles_phi_ = nullptr;
    T* twiddles_z_ = nullptr;

    static sycl::queue& require_in_order(sycl::queue& queue) {
        if (!queue.has_property<sycl::property::queue::in_order>()) {
            throw std::invalid_argument(
                "SYCL spectral filter requires an in-order queue");
        }
        return queue;
    }

    static bool supported_fft_size(int size) {
        return fft_sycl::is_power_of_two(size)
            && size >= 4 && size <= 256;
    }

    template<typename U>
    U* allocate(std::size_t size) {
        U* result = sycl::malloc_shared<U>(std::max<std::size_t>(size, 1),
                                           queue_);
        if (!result) {
            throw std::bad_alloc();
        }
        return result;
    }

    template<typename U>
    void free(U*& pointer) noexcept {
        if (pointer) {
            sycl::free(pointer, queue_);
            pointer = nullptr;
        }
    }

    void release() noexcept {
        free(twiddles_z_);
        free(twiddles_phi_);
        free(coordinates_after_);
        free(coordinates_before_);
        free(coordinate_rhs_);
        free(remaining_values_);
        free(removed_values_);
        free(block_values_);
        free(inverse_gram_);
        free(left_basis_);
        free(right_basis_);
        free(packed_indices_);
        free(block_info_);
        free(reference_);
        free(original_fourier_);
        free(packed_fourier_);
        free(temporary_);
        free(original_physical_);
        free(physical_);
    }

    static std::vector<int> phase_indices(int frequency, int size) {
        if (frequency == 0 || 2*frequency == size) {
            return {frequency};
        }
        return {frequency, size-frequency};
    }

    int state_index_from_radial(int radial_index, int i, int k) const {
        if (radial_index < layout_.v_radial_offset) {
            return layout_.u_offset+(i*layout_.nz+k)*(layout_.nr-1)
                +radial_index;
        }
        if (radial_index < layout_.w_radial_offset) {
            return layout_.v_offset+(i*layout_.nz+k)*layout_.nr
                +radial_index-layout_.v_radial_offset;
        }
        if (radial_index < layout_.p_radial_offset) {
            return layout_.w_offset+(i*layout_.nz+k)*layout_.nr
                +radial_index-layout_.w_radial_offset;
        }
        return layout_.p_offset+(i*layout_.nz+k)*layout_.nr
            +radial_index-layout_.p_radial_offset;
    }

    void validate_block(const BlockProjector& block) const {
        if (block.m() < 0 || block.m() > layout_.nphi/2
            || block.l() < 0 || block.l() > layout_.nz/2) {
            throw std::invalid_argument(
                "spectral projector block is outside the SYCL filter grid");
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
                "spectral projector layout does not match SYCL filter grid");
        }
    }

    void build_projector_data() {
        int value_offset = 0;
        int coordinate_offset = 0;
        int basis_offset = 0;
        int inverse_offset = 0;
        for (const auto& block : projector_.blocks()) {
            validate_block(block);
            KernelBlockInfo info{
                block.m(), block.l(), block.block_size(), block.dimension(),
                static_cast<int>(packed_indices_host_.size()), value_offset,
                basis_offset, inverse_offset, coordinate_offset,
                block.pressure_gauge_fixed() ? 1 : 0};

            const auto phi = phase_indices(block.m(), layout_.nphi);
            const auto z = phase_indices(block.l(), layout_.nz);
            for (int i : phi) {
                for (int k : z) {
                    for (int radial = 0; radial < layout_.radial_size;
                         ++radial) {
                        if (block.pressure_gauge_fixed()
                            && radial == layout_.radial_size-1) {
                            continue;
                        }
                        packed_indices_host_.push_back(
                            state_index_from_radial(radial, i, k));
                    }
                }
            }
            if (static_cast<int>(packed_indices_host_.size())
                    != info.index_offset+info.block_size) {
                throw std::logic_error("invalid SYCL block index count");
            }

            for (const auto& basis : block.right_basis()) {
                right_basis_host_.insert(
                    right_basis_host_.end(), basis.begin(), basis.end());
            }
            for (const auto& basis : block.left_basis()) {
                left_basis_host_.insert(
                    left_basis_host_.end(), basis.begin(), basis.end());
            }
            inverse_gram_host_.insert(
                inverse_gram_host_.end(), block.inverse_gram().begin(),
                block.inverse_gram().end());
            blocks_.push_back(info);
            value_offset += block.block_size();
            coordinate_offset += block.dimension();
            basis_offset += block.block_size()*block.dimension();
            inverse_offset += block.dimension()*block.dimension();
        }
        total_block_values_ = value_offset;
        total_coordinates_ = coordinate_offset;
        if (right_basis_host_.size() != left_basis_host_.size()
            || static_cast<int>(right_basis_host_.size()) != basis_offset
            || static_cast<int>(inverse_gram_host_.size()) != inverse_offset) {
            throw std::logic_error("invalid flattened SYCL projector basis");
        }
    }

    void ensure_reference(const std::vector<T>& reference) {
        if (static_cast<int>(reference.size()) != layout_.state_size) {
            throw std::invalid_argument(
                "reference state has the wrong SYCL filter size");
        }
        queue_.memcpy(reference_, reference.data(),
                      static_cast<std::size_t>(layout_.state_size)*sizeof(T));
    }

    template<typename Access>
    void pack_component(Access access, int offset, int radial_size,
                        int first_radial) {
        const int nphi = layout_.nphi;
        const int nz = layout_.nz;
        T* physical = physical_;
        const T* reference = reference_;
        queue_.parallel_for(
            sycl::range<3>(static_cast<std::size_t>(nphi),
                           static_cast<std::size_t>(nz),
                           static_cast<std::size_t>(radial_size)),
            [=](sycl::id<3> id) {
                const int i = static_cast<int>(id[0]);
                const int k = static_cast<int>(id[1]);
                const int j = static_cast<int>(id[2]);
                const int index = offset+(i*nz+k)*radial_size+j;
                physical[index] = access(i, k, first_radial+j)
                    -reference[index];
            });
    }

    template<typename Access>
    void unpack_component(Access access, int offset, int radial_size,
                          int first_radial) {
        const int nphi = layout_.nphi;
        const int nz = layout_.nz;
        const T* physical = physical_;
        const T* reference = reference_;
        queue_.parallel_for(
            sycl::range<3>(static_cast<std::size_t>(nphi),
                           static_cast<std::size_t>(nz),
                           static_cast<std::size_t>(radial_size)),
            [=](sycl::id<3> id) {
                const int i = static_cast<int>(id[0]);
                const int k = static_cast<int>(id[1]);
                const int j = static_cast<int>(id[2]);
                const int index = offset+(i*nz+k)*radial_size+j;
                access(i, k, first_radial+j) =
                    reference[index]+physical[index];
            });
    }

    template<typename Task>
    void pack_difference(Task& state) {
        pack_component(state.ua(), layout_.u_offset, layout_.nr-1, 1);
        pack_component(state.va(), layout_.v_offset, layout_.nr, 1);
        pack_component(state.wa(), layout_.w_offset, layout_.nr, 1);
        pack_component(state.pa(), layout_.p_offset, layout_.nr, 1);
        queue_.memcpy(original_physical_, physical_,
                      static_cast<std::size_t>(layout_.state_size)*sizeof(T));
    }

    template<typename Task>
    void unpack_sum(Task& state) {
        unpack_component(state.ua(), layout_.u_offset, layout_.nr-1, 1);
        unpack_component(state.va(), layout_.v_offset, layout_.nr, 1);
        unpack_component(state.wa(), layout_.w_offset, layout_.nr, 1);
        unpack_component(state.pa(), layout_.p_offset, layout_.nr, 1);
        state.apply_boundary_conditions();
    }

    template<bool Forward>
    void transform(T* output, const T* input, const T* twiddles, T scale,
                   int size, int lines, int stride, int inner, int outer) {
#define FDM_SYCL_FILTER_FFT_CASE(n)                                           \
        case n:                                                               \
            if constexpr (Forward) {                                          \
                fft_sycl::real_forward<n>(queue_, output, input, twiddles,    \
                                          scale, lines, stride, inner, outer);\
            } else {                                                          \
                fft_sycl::real_inverse<n>(queue_, output, input, twiddles,    \
                                          scale, lines, stride, inner, outer);\
            }                                                                 \
            break
        switch (size) {
        FDM_SYCL_FILTER_FFT_CASE(4);
        FDM_SYCL_FILTER_FFT_CASE(8);
        FDM_SYCL_FILTER_FFT_CASE(16);
        FDM_SYCL_FILTER_FFT_CASE(32);
        FDM_SYCL_FILTER_FFT_CASE(64);
        FDM_SYCL_FILTER_FFT_CASE(128);
        FDM_SYCL_FILTER_FFT_CASE(256);
        default:
            throw std::logic_error("unsupported SYCL filter FFT size");
        }
#undef FDM_SYCL_FILTER_FFT_CASE
    }

    void analysis_component(int offset, int radial_size) {
        transform<true>(
            temporary_+offset, physical_+offset, twiddles_phi_,
            T(2)/layout_.nphi, layout_.nphi, layout_.nz*radial_size,
            layout_.nz*radial_size, radial_size, radial_size);
        transform<true>(
            packed_fourier_+offset, temporary_+offset, twiddles_z_,
            T(2)/layout_.nz, layout_.nz, layout_.nphi*radial_size,
            radial_size, radial_size, layout_.nz*radial_size);
    }

    void synthesis_component(int offset, int radial_size) {
        transform<false>(
            temporary_+offset, packed_fourier_+offset, twiddles_z_, T(1),
            layout_.nz, layout_.nphi*radial_size, radial_size, radial_size,
            layout_.nz*radial_size);
        transform<false>(
            physical_+offset, temporary_+offset, twiddles_phi_, T(1),
            layout_.nphi, layout_.nz*radial_size,
            layout_.nz*radial_size, radial_size, radial_size);
    }

    void analysis() {
        analysis_component(layout_.u_offset, layout_.nr-1);
        analysis_component(layout_.v_offset, layout_.nr);
        analysis_component(layout_.w_offset, layout_.nr);
        analysis_component(layout_.p_offset, layout_.nr);
    }

    void synthesis() {
        synthesis_component(layout_.u_offset, layout_.nr-1);
        synthesis_component(layout_.v_offset, layout_.nr);
        synthesis_component(layout_.w_offset, layout_.nr);
        synthesis_component(layout_.p_offset, layout_.nr);
    }

    template<typename Geometry>
    void canonicalize_pressure_gauge(const Geometry& geometry) {
        T* fourier = packed_fourier_;
        const int p_offset = layout_.p_offset;
        const int nr = layout_.nr;
        const T r0 = static_cast<T>(geometry.r0);
        const T dr = static_cast<T>(geometry.dr);
        queue_.single_task([=]() {
            T weighted_sum = T(0);
            T weight = T(0);
            for (int j = 0; j < nr; ++j) {
                const T radius = r0+(T(j)+T(0.5))*dr;
                weighted_sum += radius*fourier[p_offset+j];
                weight += radius;
            }
            const T mean = weighted_sum/weight;
            for (int j = 0; j < nr; ++j) {
                fourier[p_offset+j] -= mean;
            }
        });
    }

    template<typename Geometry>
    void project_blocks(const Geometry& geometry,
                        NSCylSpectralRemoval removal) {
        const KernelBlockInfo* infos = block_info_;
        const int* indices = packed_indices_;
        const T* right = right_basis_;
        const T* left = left_basis_;
        const T* inverse = inverse_gram_;
        T* values = block_values_;
        T* removed_values = removed_values_;
        T* remaining_values = remaining_values_;
        T* rhs = coordinate_rhs_;
        T* before = coordinates_before_;
        T* after = coordinates_after_;
        T* fourier = packed_fourier_;
        const int nr = layout_.nr;
        const int p_offset = layout_.p_offset;
        const int p_radial_offset = layout_.p_radial_offset;
        const T r0 = static_cast<T>(geometry.r0);
        const T dr = static_cast<T>(geometry.dr);
        const bool whole = removal == NSCylSpectralRemoval::whole_fourier_blocks;
        queue_.parallel_for(
            sycl::range<1>(blocks_.size()), [=](sycl::id<1> id) {
                const KernelBlockInfo info = infos[id[0]];
                T* block = values+info.value_offset;
                T* removed = removed_values+info.value_offset;
                T* remaining = remaining_values+info.value_offset;
                T* coordinates = before+info.coordinate_offset;
                T* filtered_coordinates = after+info.coordinate_offset;
                const int* block_indices = indices+info.index_offset;
                const T* block_right = right+info.basis_offset;
                const T* block_left = left+info.basis_offset;
                const T* block_inverse = inverse+info.inverse_offset;
                T* block_rhs = rhs+info.coordinate_offset;

                for (int row = 0; row < info.block_size; ++row) {
                    block[row] = fourier[block_indices[row]];
                }
                for (int coordinate = 0; coordinate < info.dimension;
                     ++coordinate) {
                    T value = T(0);
                    T compensation = T(0);
                    for (int row = 0; row < info.block_size; ++row) {
                        const T product =
                            block_left[coordinate*info.block_size+row]
                            *block[row]-compensation;
                        const T next = value+product;
                        compensation = (next-value)-product;
                        value = next;
                    }
                    block_rhs[coordinate] = value;
                }
                for (int row = 0; row < info.dimension; ++row) {
                    T value = T(0);
                    for (int column = 0; column < info.dimension; ++column) {
                        value += block_inverse[row*info.dimension+column]
                            *block_rhs[column];
                    }
                    coordinates[row] = value;
                }
                for (int row = 0; row < info.block_size; ++row) {
                    T value = block[row];
                    if (!whole) {
                        value = T(0);
                        for (int coordinate = 0;
                             coordinate < info.dimension; ++coordinate) {
                            value += block_right[
                                coordinate*info.block_size+row]
                                *coordinates[coordinate];
                        }
                    }
                    removed[row] = value;
                    block[row] -= value;
                    fourier[block_indices[row]] = block[row];
                }
                if (info.pressure_gauge_fixed) {
                    T weighted_sum = T(0);
                    for (int j = 0; j < nr-1; ++j) {
                        const T radius = r0+(T(j)+T(0.5))*dr;
                        weighted_sum += radius
                            *block[p_radial_offset+j];
                    }
                    const T last_radius = r0+(T(nr)-T(0.5))*dr;
                    fourier[p_offset+nr-1] = -weighted_sum/last_radius;
                }
                for (int coordinate = 0; coordinate < info.dimension;
                     ++coordinate) {
                    T value = T(0);
                    T compensation = T(0);
                    for (int row = 0; row < info.block_size; ++row) {
                        const T product =
                            block_left[coordinate*info.block_size+row]
                            *block[row]-compensation;
                        const T next = value+product;
                        compensation = (next-value)-product;
                        value = next;
                    }
                    block_rhs[coordinate] = value;
                }
                for (int row = 0; row < info.dimension; ++row) {
                    T value = T(0);
                    for (int column = 0; column < info.dimension; ++column) {
                        value += block_inverse[row*info.dimension+column]
                            *block_rhs[column];
                    }
                    filtered_coordinates[row] = value;
                }
                // In exact arithmetic the oblique projector is idempotent.
                // A second application removes only the residual left by the
                // long single-precision dot products on the device.
                if (!whole) {
                    for (int row = 0; row < info.block_size; ++row) {
                        T correction = T(0);
                        for (int coordinate = 0;
                             coordinate < info.dimension; ++coordinate) {
                            correction += block_right[
                                coordinate*info.block_size+row]
                                *filtered_coordinates[coordinate];
                        }
                        removed[row] += correction;
                        block[row] -= correction;
                        fourier[block_indices[row]] = block[row];
                    }
                    if (info.pressure_gauge_fixed) {
                        T weighted_sum = T(0);
                        for (int j = 0; j < nr-1; ++j) {
                            const T radius = r0+(T(j)+T(0.5))*dr;
                            weighted_sum += radius
                                *block[p_radial_offset+j];
                        }
                        const T last_radius = r0+(T(nr)-T(0.5))*dr;
                        fourier[p_offset+nr-1] =
                            -weighted_sum/last_radius;
                    }
                    for (int coordinate = 0;
                         coordinate < info.dimension; ++coordinate) {
                        T value = T(0);
                        T compensation = T(0);
                        for (int row = 0; row < info.block_size; ++row) {
                            const T product = block_left[
                                coordinate*info.block_size+row]
                                *block[row]-compensation;
                            const T next = value+product;
                            compensation = (next-value)-product;
                            value = next;
                        }
                        block_rhs[coordinate] = value;
                    }
                    for (int row = 0; row < info.dimension; ++row) {
                        T value = T(0);
                        for (int column = 0;
                             column < info.dimension; ++column) {
                            value += block_inverse[
                                row*info.dimension+column]
                                *block_rhs[column];
                        }
                        filtered_coordinates[row] = value;
                    }
                }
                for (int row = 0; row < info.block_size; ++row) {
                    T value = T(0);
                    for (int coordinate = 0;
                         coordinate < info.dimension; ++coordinate) {
                        value += block_right[
                            coordinate*info.block_size+row]
                            *filtered_coordinates[coordinate];
                    }
                    remaining[row] = value;
                }
            });
    }

    static double norm(const T* values, int size) {
        long double result = 0;
        for (int index = 0; index < size; ++index) {
            const long double value = values[index];
            result += value*value;
        }
        return std::sqrt(static_cast<double>(result));
    }

    template<typename Geometry>
    NSCylSpectralFilterDiagnostics diagnostics(
        const Geometry& geometry) const {
        NSCylSpectralFilterDiagnostics result;
        result.velocity_perturbation_norm =
            layout_.velocity_norm(geometry, original_physical_);
        result.filtered_velocity_norm =
            layout_.velocity_norm(geometry, physical_);
        result.packed_perturbation_norm =
            norm(original_fourier_, layout_.state_size);

        long double removed_velocity_squared = 0;
        const long double measure = static_cast<long double>(geometry.dr)
            *geometry.dphi*geometry.dz;
        for (int i = 0; i < layout_.nphi; ++i) {
            for (int k = 0; k < layout_.nz; ++k) {
                for (int j = 1; j < layout_.nr; ++j) {
                    const int index = layout_.u_offset
                        +(i*layout_.nz+k)*(layout_.nr-1)+j-1;
                    const long double difference =
                        original_physical_[index]-physical_[index];
                    removed_velocity_squared +=
                        (geometry.r0+j*geometry.dr)*difference*difference;
                }
                for (int j = 1; j <= layout_.nr; ++j) {
                    const long double radius =
                        geometry.r0+(j-0.5L)*geometry.dr;
                    const int plane = (i*layout_.nz+k)*layout_.nr+j-1;
                    for (int offset : {layout_.v_offset, layout_.w_offset}) {
                        const int index = offset+plane;
                        const long double difference =
                            original_physical_[index]-physical_[index];
                        removed_velocity_squared +=
                            radius*difference*difference;
                    }
                }
            }
        }
        result.removed_velocity_norm = std::sqrt(static_cast<double>(
            measure*removed_velocity_squared));

        long double removed_squared = 0;
        long double remaining_squared = 0;
        for (std::size_t block_index = 0; block_index < blocks_.size();
             ++block_index) {
            const auto& info = blocks_[block_index];
            NSCylSpectralBlockFilterDiagnostics block;
            block.m = info.m;
            block.l = info.l;
            long double block_squared = 0;
            for (int row = 0; row < info.block_size; ++row) {
                const int index = info.value_offset+row;
                const long double removed = removed_values_[index];
                const long double filtered = block_values_[index];
                block_squared += (removed+filtered)*(removed+filtered);
            }
            block.block_norm = std::sqrt(static_cast<double>(block_squared));
            block.removed_norm = norm(
                removed_values_+info.value_offset, info.block_size);
            block.remaining_unstable_norm = norm(
                remaining_values_+info.value_offset, info.block_size);
            block.coordinates_before.reserve(info.dimension);
            block.coordinates_after.reserve(info.dimension);
            for (int coordinate = 0; coordinate < info.dimension;
                 ++coordinate) {
                block.coordinates_before.push_back(coordinates_before_[
                    info.coordinate_offset+coordinate]);
                block.coordinates_after.push_back(coordinates_after_[
                    info.coordinate_offset+coordinate]);
            }
            removed_squared += block.removed_norm*block.removed_norm;
            remaining_squared += block.remaining_unstable_norm
                *block.remaining_unstable_norm;
            result.blocks.push_back(std::move(block));
        }
        result.removed_norm = std::sqrt(static_cast<double>(removed_squared));
        result.remaining_unstable_norm =
            std::sqrt(static_cast<double>(remaining_squared));
        return result;
    }

    template<typename Task>
    NSCylSpectralFilterDiagnostics execute(
        Task& state, const std::vector<T>& reference,
        NSCylSpectralRemoval removal, bool update_state) {
        if (state.nr != layout_.nr || state.nphi != layout_.nphi
            || state.nz != layout_.nz) {
            throw std::invalid_argument(
                "NSCyl state dimensions do not match SYCL spectral filter");
        }
        if (blocks_.empty()) {
            return {};
        }
        ensure_reference(reference);
        pack_difference(state);
        analysis();
        canonicalize_pressure_gauge(state);
        queue_.memcpy(original_fourier_, packed_fourier_,
                      static_cast<std::size_t>(layout_.state_size)*sizeof(T));
        project_blocks(state, removal);
        synthesis();
        if (update_state) {
            unpack_sum(state);
        } else {
            queue_.wait_and_throw();
        }
        return diagnostics(state);
    }
};

} // namespace fdm
