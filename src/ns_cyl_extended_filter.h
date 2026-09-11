#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "asp_gauss.h"
#include "config.h"
#include "ns_cyl_fourier_block.h"
#include "ns_cyl_fourier_native.h"
#include "ns_cyl_spectral_projector.h"
#include "ns_cyl_state.h"
#include "projection.h"

extern "C" {
void sgesv_(int* n, int* nrhs, float* matrix, int* lda, int* pivots,
            float* right_hand_side, int* ldb, int* info);
}

namespace fdm {

struct NSCylExtendedBlockFilterDiagnostics {
    int m = -1;
    int l = -1;
    int continuation_dimension = 0;
    double response_norm = 0;
    double inverse_response_norm = 0;
    double response_condition = 0;
    double unstable_coordinate_norm_before = 0;
    double unstable_coordinate_norm_after = 0;
    double coefficient_norm = 0;
    double correction_velocity_norm = 0;
    double boundary_rms = 0;
    double boundary_maximum = 0;
    double coordinate_to_correction_gain = 0;
    double coordinate_to_boundary_rms_gain = 0;
};

struct NSCylExtendedFilterDiagnostics {
    double unstable_coordinate_norm_before = 0;
    double unstable_coordinate_norm_after = 0;
    double correction_velocity_norm = 0;
    double original_domain_change_norm = 0;
    std::vector<NSCylExtendedBlockFilterDiagnostics> blocks;
};

template<typename T>
struct NSCylOuterBoundaryVelocity {
    int nphi = 0;
    int nz = 0;
    std::vector<T> radial;
    std::vector<T> axial;
    std::vector<T> azimuthal;

    double rms_norm() const {
        if (radial.empty()) {
            return 0;
        }
        long double sum = 0;
        for (std::size_t index = 0; index < radial.size(); ++index) {
            const long double ur = radial[index];
            const long double uz = axial[index];
            const long double uphi = azimuthal[index];
            sum += ur*ur+uz*uz+uphi*uphi;
        }
        return std::sqrt(static_cast<double>(sum/radial.size()));
    }

    double maximum_norm() const {
        double result = 0;
        for (std::size_t index = 0; index < radial.size(); ++index) {
            result = std::max(result, std::sqrt(
                static_cast<double>(radial[index])*radial[index]
                +static_cast<double>(axial[index])*axial[index]
                +static_cast<double>(azimuthal[index])*azimuthal[index]));
        }
        return result;
    }
};

// Builds corrections supported in the auxiliary annulus.  Each restricted
// right mode is multiplied by a small radial polynomial basis and passed
// through a stationary discrete Stokes inverse.  If C denotes the unstable
// coordinate functionals and W the resulting continuations, M=CW.  The
// default uses the square response M^{-1}; an expanded basis uses the
// minimum-trace right inverse H^{-1}M^T(MH^{-1}M^T)^{-1}.  Thus C(q+Wa)=0
// remains exact while every velocity degree of freedom in the original
// cylinder is unchanged.
template<typename T>
class NSCylExtendedSpectralFilter {
public:
    using Layout = NSCylStateLayout<T>;
    using Component = typename Layout::Component;
    using Projector = NSCylSpectralProjector<T>;
    using BlockProjector = NSCylSpectralBlockProjector<T>;

    struct Geometry {
        int nr = 0;
        int nphi = 0;
        int nz = 0;
        double r0 = 0;
        double R = 0;
        double h1 = 0;
        double h2 = 0;
        double dr = 0;
        double dphi = 0;
        double dz = 0;
        double Re = 0;
        double dt = 0;
        double U0 = 0;
    };

    NSCylExtendedSpectralFilter(const Config& config, Projector projector)
        : geometry_(read_geometry(config))
        , base_outer_radius_(config.get(
              "spectral", "base_outer_radius", geometry_.R))
        , original_nr_(aligned_original_cells(
              geometry_, base_outer_radius_))
        , auxiliary_nr_(geometry_.nr-original_nr_)
        , layout_(geometry_.nr, geometry_.nz, geometry_.nphi)
        , fft_(geometry_.nphi, geometry_.nz)
        , projector_(std::move(projector))
        , physical_(layout_.state_size)
        , original_physical_(layout_.state_size)
        , correction_physical_(layout_.state_size)
        , packed_fourier_(layout_.state_size)
        , values_(fft_.size())
        , coefficients_(fft_.size()) {
        if (auxiliary_nr_ < 2) {
            throw std::invalid_argument(
                "extended NSCyl filter needs at least two auxiliary cells");
        }
        response_condition_limit_ = config.get(
            "extended", "response_condition_limit", 1e12);
        response_regularization_ = config.get(
            "extended", "response_regularization", 0.0);
        response_basis_count_ = config.get(
            "extended", "response_basis_count", 1);
        response_trace_horizon_steps_ = config.get(
            "extended", "response_trace_horizon_steps", 0);
        response_trace_sample_stride_ = config.get(
            "extended", "response_trace_sample_stride", 1);
        response_cost_ridge_ = config.get(
            "extended", "response_cost_ridge", 0.0);
        if (!(response_condition_limit_ >= 1)) {
            throw std::invalid_argument(
                "extended response condition limit must be at least one");
        }
        if (!(response_regularization_ >= 0)
            || !std::isfinite(response_regularization_)) {
            throw std::invalid_argument(
                "extended response regularization must be finite and "
                "nonnegative");
        }
        if (response_basis_count_ <= 0 || response_basis_count_ > 4
            || response_trace_horizon_steps_ < 0
            || response_trace_sample_stride_ <= 0
            || !(response_cost_ridge_ >= 0)
            || !std::isfinite(response_cost_ridge_)) {
            throw std::invalid_argument(
                "invalid extended response basis or trace-cost settings");
        }
        validate_projector();
        build_corrections();
    }

    const Geometry& geometry() const { return geometry_; }
    int original_nr() const { return original_nr_; }
    int auxiliary_nr() const { return auxiliary_nr_; }
    double base_outer_radius() const { return base_outer_radius_; }

    // Embed the independent unknowns from Omega into G.  The auxiliary
    // degrees of freedom are zero; the interface-normal velocity is an
    // independent interior face of G and is therefore also initialized to
    // zero.  This embedding preserves the packed pressure convention.
    std::vector<T> embed_original_perturbation(
        const std::vector<T>& original) const {
        const Layout source(original_nr_, geometry_.nz, geometry_.nphi);
        if (static_cast<int>(original.size()) != source.state_size) {
            throw std::invalid_argument(
                "original perturbation has the wrong packed size");
        }
        std::vector<T> result(layout_.state_size, T(0));
        for (int i = 0; i < geometry_.nphi; ++i) {
            for (int k = 0; k < geometry_.nz; ++k) {
                for (int j = 1; j < original_nr_; ++j) {
                    result[state_index(Component::u, i, k, j)] = original[
                        packed_index(source, Component::u, i, k, j)];
                }
                for (Component component : {
                        Component::v, Component::w, Component::p}) {
                    for (int j = 1; j <= original_nr_; ++j) {
                        result[state_index(component, i, k, j)] = original[
                            packed_index(source, component, i, k, j)];
                    }
                }
            }
        }
        return result;
    }

    // Return the wall velocity induced by the most recent supported
    // correction Wc.  For cell-centred tangential velocities the trace is
    // the average of the samples adjacent to r=r_o.  The correction is zero
    // in Omega, so this is one half of the first auxiliary sample.  Using the
    // trace of the complete extended state here would incorrectly mix the
    // last interior sample into the control.
    NSCylOuterBoundaryVelocity<T> correction_boundary_velocity() const {
        NSCylOuterBoundaryVelocity<T> result;
        result.nphi = geometry_.nphi;
        result.nz = geometry_.nz;
        const std::size_t size =
            static_cast<std::size_t>(geometry_.nphi)*geometry_.nz;
        result.radial.resize(size);
        result.axial.resize(size);
        result.azimuthal.resize(size);
        for (int i = 0; i < geometry_.nphi; ++i) {
            for (int k = 0; k < geometry_.nz; ++k) {
                const std::size_t plane =
                    static_cast<std::size_t>(i)*geometry_.nz+k;
                result.radial[plane] = correction_physical_[state_index(
                    Component::u, i, k, original_nr_)];
                result.axial[plane] = T(0.5)*(
                    correction_physical_[state_index(
                        Component::v, i, k, original_nr_)]
                    +correction_physical_[state_index(
                        Component::v, i, k, original_nr_+1)]);
                result.azimuthal[plane] = T(0.5)*(
                    correction_physical_[state_index(
                        Component::w, i, k, original_nr_)]
                    +correction_physical_[state_index(
                        Component::w, i, k, original_nr_+1)]);
            }
        }
        return result;
    }

    // Adjust only omega so that the result has the unstable coordinates of
    // target.  This is the supported analogue of projection onto a nonlinear
    // graph: apply the linear correction to q-target, then add target back.
    NSCylExtendedFilterDiagnostics apply_towards(
        std::vector<T>& extended_perturbation,
        const std::vector<T>& target) {
        if (extended_perturbation.size() != target.size()
            || static_cast<int>(target.size()) != layout_.state_size) {
            throw std::invalid_argument(
                "extended target has the wrong packed size");
        }
        std::vector<T> difference(target.size());
        for (std::size_t index = 0; index < target.size(); ++index) {
            difference[index] = extended_perturbation[index]-target[index];
        }
        auto result = apply(difference);
        for (std::size_t index = 0; index < target.size(); ++index) {
            extended_perturbation[index] = target[index]+difference[index];
        }
        return result;
    }

    NSCylExtendedFilterDiagnostics apply(
        std::vector<T>& extended_perturbation,
        bool detailed_block_diagnostics = false) {
        if (static_cast<int>(extended_perturbation.size())
            != layout_.state_size) {
            throw std::invalid_argument(
                "extended perturbation has the wrong packed size");
        }

        physical_ = extended_perturbation;
        original_physical_ = physical_;
        analysis();
        canonicalize_pressure_gauge();

        NSCylExtendedFilterDiagnostics result;
        long double before2 = 0;
        long double after2 = 0;
        for (std::size_t index = 0;
             index < projector_.blocks().size(); ++index) {
            const auto& projector = projector_.blocks()[index];
            const auto& correction = corrections_[index];
            gather_block(projector);

            const int dimension = projector.dimension();
            std::vector<T> before(dimension, T(0));
            std::vector<T> after(dimension, T(0));
            std::vector<T> amplitudes(correction.basis.size(), T(0));
            std::vector<T> block_correction;
            if (detailed_block_diagnostics) {
                block_correction.assign(projector.block_size(), T(0));
            }
            projector.coordinates(before.data(), block_.data());
            compute_amplitudes(correction, before, amplitudes);
            for (int column = 0;
                 column < static_cast<int>(correction.basis.size());
                 ++column) {
                const auto& basis = correction.basis[column];
                for (int coordinate = 0;
                     coordinate < projector.block_size(); ++coordinate) {
                    const T value = amplitudes[column]*basis[coordinate];
                    block_[coordinate] += value;
                    if (detailed_block_diagnostics) {
                        block_correction[coordinate] += value;
                    }
                }
            }
            projector.coordinates(after.data(), block_.data());
            scatter_block(projector, block_.data());

            NSCylExtendedBlockFilterDiagnostics block_result;
            block_result.m = projector.m();
            block_result.l = projector.l();
            block_result.continuation_dimension =
                static_cast<int>(correction.basis.size());
            block_result.response_norm = correction.response_norm;
            block_result.inverse_response_norm =
                correction.inverse_response_norm;
            block_result.response_condition = correction.response_condition;
            for (int coordinate = 0; coordinate < dimension; ++coordinate) {
                const long double b = before[coordinate];
                const long double a = after[coordinate];
                block_result.unstable_coordinate_norm_before +=
                    static_cast<double>(b*b);
                block_result.unstable_coordinate_norm_after +=
                    static_cast<double>(a*a);
            }
            for (T amplitude : amplitudes) {
                const long double value = amplitude;
                block_result.coefficient_norm +=
                    static_cast<double>(value*value);
            }
            block_result.unstable_coordinate_norm_before = std::sqrt(
                block_result.unstable_coordinate_norm_before);
            block_result.unstable_coordinate_norm_after = std::sqrt(
                block_result.unstable_coordinate_norm_after);
            block_result.coefficient_norm = std::sqrt(
                block_result.coefficient_norm);
            if (detailed_block_diagnostics) {
                const auto metrics = physical_block_metrics(
                    projector, block_correction);
                block_result.correction_velocity_norm =
                    metrics.velocity_norm;
                block_result.boundary_rms = metrics.boundary_rms;
                block_result.boundary_maximum = metrics.boundary_maximum;
                if (block_result.unstable_coordinate_norm_before > 0) {
                    block_result.coordinate_to_correction_gain =
                        metrics.velocity_norm
                        /block_result.unstable_coordinate_norm_before;
                    block_result.coordinate_to_boundary_rms_gain =
                        metrics.boundary_rms
                        /block_result.unstable_coordinate_norm_before;
                }
            }
            before2 += block_result.unstable_coordinate_norm_before
                *block_result.unstable_coordinate_norm_before;
            after2 += block_result.unstable_coordinate_norm_after
                *block_result.unstable_coordinate_norm_after;
            result.blocks.push_back(block_result);
        }

        synthesis();
        for (int i = 0; i < layout_.state_size; ++i) {
            correction_physical_[i] = physical_[i]-original_physical_[i];
        }
        result.unstable_coordinate_norm_before = std::sqrt(
            static_cast<double>(before2));
        result.unstable_coordinate_norm_after = std::sqrt(
            static_cast<double>(after2));
        result.correction_velocity_norm = layout_.velocity_norm(
            geometry_, correction_physical_.data());
        result.original_domain_change_norm =
            original_domain_velocity_norm(correction_physical_.data());
        extended_perturbation = physical_;
        return result;
    }

private:
    struct BlockPhysicalMetrics {
        double velocity_norm = 0;
        double boundary_rms = 0;
        double boundary_maximum = 0;
    };

    struct CorrectionBlock {
        std::vector<std::vector<T>> basis;
        std::vector<T> response;
        std::vector<T> inverse_response;
        std::vector<T> trace_gram;
        double response_norm = 0;
        double inverse_response_norm = 0;
        double response_condition = 0;
    };

    Geometry geometry_;
    double base_outer_radius_;
    int original_nr_;
    int auxiliary_nr_;
    Layout layout_;
    PeriodicPackedFFT2<T> fft_;
    Projector projector_;
    double response_condition_limit_ = 0;
    double response_regularization_ = 0;
    int response_basis_count_ = 1;
    int response_trace_horizon_steps_ = 0;
    int response_trace_sample_stride_ = 1;
    double response_cost_ridge_ = 0;
    std::vector<CorrectionBlock> corrections_;
    std::vector<T> physical_;
    std::vector<T> original_physical_;
    std::vector<T> correction_physical_;
    std::vector<T> packed_fourier_;
    std::vector<T> values_;
    std::vector<T> coefficients_;
    std::vector<T> full_block_;
    std::vector<T> block_;

    static Geometry read_geometry(const Config& config) {
        Geometry result;
        result.nr = config.get("ns", "nr", 32);
        result.nphi = config.get("ns", "nphi", 32);
        result.nz = config.get("ns", "nz", 32);
        result.r0 = config.get("ns", "r", M_PI/2);
        result.R = config.get("ns", "R", M_PI);
        result.h1 = config.get("ns", "h1", 0.0);
        result.h2 = config.get("ns", "h2", 10.0);
        result.Re = config.get("ns", "Re", 1.0);
        result.dt = config.get("ns", "dt", 0.001);
        result.U0 = config.get("ns", "u0", 1.0);
        if (result.nr < 4 || result.nphi <= 0 || result.nz <= 0
            || result.nphi%2 != 0 || result.nz%2 != 0
            || !(result.R > result.r0) || !(result.h2 > result.h1)
            || !(result.Re > 0) || !(result.dt > 0)) {
            throw std::invalid_argument("invalid extended NSCyl geometry");
        }
        result.dr = (result.R-result.r0)/result.nr;
        result.dphi = 2*M_PI/result.nphi;
        result.dz = (result.h2-result.h1)/result.nz;
        return result;
    }

    static int aligned_original_cells(const Geometry& geometry,
                                      double outer_radius) {
        const double cells = (outer_radius-geometry.r0)/geometry.dr;
        const int rounded = static_cast<int>(std::llround(cells));
        const double tolerance = 128*std::numeric_limits<double>::epsilon()
            *std::max({1.0, std::abs(cells), std::abs(outer_radius)});
        if (rounded < 2 || rounded >= geometry.nr
            || std::abs(cells-rounded) > tolerance) {
            throw std::invalid_argument(
                "base_outer_radius must be an interior radial grid face");
        }
        return rounded;
    }

    static std::string number(double value) {
        std::ostringstream output;
        output << std::setprecision(17) << value;
        return output.str();
    }

    Config auxiliary_config(double reynolds) const {
        Config result;
        std::vector<std::string> arguments = {
            "ns_cyl_extended_filter",
            "--ns:r="+number(base_outer_radius_),
            "--ns:R="+number(geometry_.R),
            "--ns:h1="+number(geometry_.h1),
            "--ns:h2="+number(geometry_.h2),
            "--ns:u0=0",
            "--ns:Re="+number(reynolds),
            "--ns:dt="+number(geometry_.dt),
            "--ns:nr="+std::to_string(auxiliary_nr_),
            "--ns:nphi="+std::to_string(geometry_.nphi),
            "--ns:nz="+std::to_string(geometry_.nz),
            "--ns:verbose=0"
        };
        std::vector<char*> argv;
        argv.reserve(arguments.size());
        for (auto& argument : arguments) {
            argv.push_back(argument.data());
        }
        result.rewrite(static_cast<int>(argv.size()), argv.data());
        return result;
    }

    Config extended_dynamics_config() const {
        Config result;
        std::vector<std::string> arguments = {
            "ns_cyl_extended_filter",
            "--ns:r="+number(geometry_.r0),
            "--ns:R="+number(geometry_.R),
            "--ns:h1="+number(geometry_.h1),
            "--ns:h2="+number(geometry_.h2),
            "--ns:u0="+number(geometry_.U0),
            "--ns:Re="+number(geometry_.Re),
            "--ns:dt="+number(geometry_.dt),
            "--ns:nr="+std::to_string(geometry_.nr),
            "--ns:nphi="+std::to_string(geometry_.nphi),
            "--ns:nz="+std::to_string(geometry_.nz),
            "--ns:verbose=0",
            "--spectral:base_outer_radius="+number(base_outer_radius_)
        };
        std::vector<char*> argv;
        argv.reserve(arguments.size());
        for (auto& argument : arguments) {
            argv.push_back(argument.data());
        }
        result.rewrite(static_cast<int>(argv.size()), argv.data());
        return result;
    }

    static void solve_dense(std::vector<T>& matrix,
                            std::vector<T>& right_hand_sides,
                            int size, int right_hand_side_count) {
        int leading_dimension = size;
        int info = 0;
        std::vector<int> pivots(size);
        if constexpr (std::is_same_v<T, double>) {
            dgesv_(&size, &right_hand_side_count, matrix.data(),
                   &leading_dimension, pivots.data(),
                   right_hand_sides.data(), &leading_dimension, &info);
        } else {
            static_assert(std::is_same_v<T, float>);
            sgesv_(&size, &right_hand_side_count, matrix.data(),
                   &leading_dimension, pivots.data(),
                   right_hand_sides.data(), &leading_dimension, &info);
        }
        if (info != 0) {
            throw std::runtime_error(
                "auxiliary Stokes solve failed, LAPACK info="
                +std::to_string(info));
        }
    }

    void validate_projector() const {
        for (const auto& block : projector_.blocks()) {
            const int phi_phases =
                (block.m() == 0 || 2*block.m() == geometry_.nphi) ? 1 : 2;
            const int z_phases =
                (block.l() == 0 || 2*block.l() == geometry_.nz) ? 1 : 2;
            const bool gauge = block.m() == 0 && block.l() == 0;
            const int expected_size = layout_.radial_size
                *phi_phases*z_phases-(gauge ? 1 : 0);
            if (block.phase_count() != phi_phases*z_phases
                || block.radial_size() != layout_.radial_size
                || block.block_size() != expected_size
                || block.pressure_gauge_fixed() != gauge) {
                throw std::invalid_argument(
                    "extended projector layout does not match G");
            }
        }
    }

    std::vector<int> local_velocity_indices(int phase_count) const {
        const Layout auxiliary(auxiliary_nr_, geometry_.nz, geometry_.nphi);
        std::vector<int> result;
        result.reserve(static_cast<std::size_t>(phase_count)
                       *(3*auxiliary_nr_-1));
        for (int phase = 0; phase < phase_count; ++phase) {
            const int offset = phase*auxiliary.radial_size;
            for (int j = 1; j < auxiliary_nr_; ++j) {
                result.push_back(offset+auxiliary.radial_index(Component::u, j));
            }
            for (Component component : {Component::v, Component::w}) {
                for (int j = 1; j <= auxiliary_nr_; ++j) {
                    result.push_back(
                        offset+auxiliary.radial_index(component, j));
                }
            }
        }
        return result;
    }

    std::vector<int> full_auxiliary_velocity_indices(int phase_count) const {
        std::vector<int> result;
        result.reserve(static_cast<std::size_t>(phase_count)
                       *(3*auxiliary_nr_-1));
        for (int phase = 0; phase < phase_count; ++phase) {
            const int offset = phase*layout_.radial_size;
            for (int j = 1; j < auxiliary_nr_; ++j) {
                result.push_back(offset+layout_.radial_index(
                    Component::u, original_nr_+j));
            }
            for (Component component : {Component::v, Component::w}) {
                for (int j = 1; j <= auxiliary_nr_; ++j) {
                    result.push_back(offset+layout_.radial_index(
                        component, original_nr_+j));
                }
            }
        }
        return result;
    }

    static T shifted_legendre(int degree, double coordinate) {
        const T x = static_cast<T>(2*coordinate-1);
        if (degree == 0) {
            return T(1);
        }
        T previous = T(1);
        T value = x;
        for (int order = 2; order <= degree; ++order) {
            const T next = (static_cast<T>(2*order-1)*x*value
                            -static_cast<T>(order-1)*previous)
                /static_cast<T>(order);
            previous = value;
            value = next;
        }
        return value;
    }

    std::vector<T> local_velocity_weights(
        int phase_count, int degree) const {
        std::vector<T> result;
        result.reserve(static_cast<std::size_t>(phase_count)
                       *(3*auxiliary_nr_-1));
        for (int phase = 0; phase < phase_count; ++phase) {
            for (int j = 1; j < auxiliary_nr_; ++j) {
                result.push_back(shifted_legendre(
                    degree, static_cast<double>(j)/auxiliary_nr_));
            }
            for (Component component : {Component::v, Component::w}) {
                (void)component;
                for (int j = 1; j <= auxiliary_nr_; ++j) {
                    result.push_back(shifted_legendre(
                        degree, (static_cast<double>(j)-0.5)/auxiliary_nr_));
                }
            }
        }
        return result;
    }

    std::vector<T> physical_boundary_trace(
        const BlockProjector& projector,
        const std::vector<T>& reduced_block) {
        if (static_cast<int>(reduced_block.size()) != projector.block_size()) {
            throw std::invalid_argument(
                "boundary trace block has the wrong packed size");
        }

        std::vector<T> expanded;
        const T* block = reduced_block.data();
        if (projector.pressure_gauge_fixed()) {
            expanded.resize(
                static_cast<std::size_t>(projector.phase_count())
                *layout_.radial_size);
            layout_.expand_zero_gauge_block(
                geometry_, reduced_block.data(), expanded.data());
            block = expanded.data();
        }

        const auto phi = packed_indices(projector.m(), layout_.nphi);
        const auto z = packed_indices(projector.l(), layout_.nz);
        std::vector<T> trace(3*fft_.size(), T(0));
        std::vector<T> plane_fourier(fft_.size(), T(0));
        std::vector<T> plane_physical(fft_.size(), T(0));
        int component_number = 0;
        for (Component component : {
                 Component::u, Component::v, Component::w}) {
            std::fill(plane_fourier.begin(), plane_fourier.end(), T(0));
            int phase = 0;
            for (int i : phi) {
                for (int k : z) {
                    const std::size_t offset =
                        static_cast<std::size_t>(phase)*layout_.radial_size;
                    if (component == Component::u) {
                        plane_fourier[plane_index(i, k)] = block[
                            offset+layout_.radial_index(
                                component, original_nr_)];
                    } else {
                        plane_fourier[plane_index(i, k)] = T(0.5)*(
                            block[offset+layout_.radial_index(
                                component, original_nr_)]
                            +block[offset+layout_.radial_index(
                                component, original_nr_+1)]);
                    }
                    ++phase;
                }
            }
            fft_.synthesis(plane_fourier.data(), plane_physical.data());
            std::copy(plane_physical.begin(), plane_physical.end(),
                      trace.begin()+component_number*fft_.size());
            ++component_number;
        }
        return trace;
    }

    void add_trace_cost_ridge(std::vector<T>& gram, int size) const {
        if (response_cost_ridge_ == 0) {
            return;
        }
        long double diagonal_sum = 0;
        for (int row = 0; row < size; ++row) {
            diagonal_sum += std::abs(static_cast<long double>(
                gram[static_cast<std::size_t>(row)*size+row]));
        }
        const long double scale = diagonal_sum > 0
            ? diagonal_sum/size : 1;
        for (int row = 0; row < size; ++row) {
            gram[static_cast<std::size_t>(row)*size+row] +=
                static_cast<T>(response_cost_ridge_*scale);
        }
    }

    std::vector<T> build_trace_gram(
        const BlockProjector& projector,
        const std::vector<std::vector<T>>& basis) {
        const int size = static_cast<int>(basis.size());
        std::vector<T> result(
            static_cast<std::size_t>(size)*size, T(0));

        // Keep the original zero-horizon calculation unchanged.  Apart from
        // avoiding unnecessary time stepping, this preserves the default
        // one-continuation path down to its floating-point operation order.
        if (response_trace_horizon_steps_ == 0) {
            std::vector<double> diagonal(size);
            for (int column = 0; column < size; ++column) {
                const auto metrics = physical_block_metrics(
                    projector, basis[column]);
                diagonal[column] =
                    metrics.boundary_rms*metrics.boundary_rms;
                result[static_cast<std::size_t>(column)*size+column] =
                    static_cast<T>(diagonal[column]);
            }
            for (int row = 0; row < size; ++row) {
                for (int column = row+1; column < size; ++column) {
                    std::vector<T> sum(projector.block_size());
                    for (int coordinate = 0;
                         coordinate < projector.block_size(); ++coordinate) {
                        sum[coordinate] = basis[row][coordinate]
                            +basis[column][coordinate];
                    }
                    const auto metrics = physical_block_metrics(
                        projector, sum);
                    const T entry = static_cast<T>(0.5*(
                        metrics.boundary_rms*metrics.boundary_rms
                        -diagonal[row]-diagonal[column]));
                    result[static_cast<std::size_t>(row)*size+column] = entry;
                    result[static_cast<std::size_t>(column)*size+row] = entry;
                }
            }
            add_trace_cost_ridge(result, size);
            return result;
        }

        NSCylFourierBlockNative<T> dynamics(
            extended_dynamics_config(), projector.m(), projector.l(), 1);
        if (dynamics.size() != projector.block_size()) {
            throw std::logic_error(
                "trace dynamics layout does not match spectral block");
        }
        std::vector<std::vector<T>> states = basis;
        std::vector<std::vector<T>> next(
            size, std::vector<T>(projector.block_size(), T(0)));
        int sample_count = 0;
        for (int step = 0; step <= response_trace_horizon_steps_; ++step) {
            if (step%response_trace_sample_stride_ == 0
                || step == response_trace_horizon_steps_) {
                std::vector<std::vector<T>> traces;
                traces.reserve(size);
                for (const auto& state : states) {
                    traces.push_back(physical_boundary_trace(
                        projector, state));
                }
                const long double normalization =
                    static_cast<long double>(layout_.nphi)*layout_.nz;
                for (int row = 0; row < size; ++row) {
                    for (int column = row; column < size; ++column) {
                        long double entry = 0;
                        for (std::size_t coordinate = 0;
                             coordinate < traces[row].size(); ++coordinate) {
                            entry += static_cast<long double>(
                                traces[row][coordinate])
                                *traces[column][coordinate];
                        }
                        result[static_cast<std::size_t>(row)*size+column] +=
                            static_cast<T>(entry/normalization);
                        if (row != column) {
                            result[static_cast<std::size_t>(column)*size+row]
                                += static_cast<T>(entry/normalization);
                        }
                    }
                }
                ++sample_count;
            }
            if (step != response_trace_horizon_steps_) {
                for (int column = 0; column < size; ++column) {
                    dynamics.apply(next[column].data(),
                                   states[column].data());
                }
                states.swap(next);
            }
        }
        for (T& entry : result) {
            entry /= static_cast<T>(sample_count);
        }
        add_trace_cost_ridge(result, size);
        return result;
    }

    std::vector<T> minimum_trace_right_inverse(
        const BlockProjector& projector, const std::vector<T>& response,
        const std::vector<T>& trace_gram, int dimension,
        int continuation_dimension) const {
        if (continuation_dimension == dimension) {
            std::vector<T> result(response.size());
            const T pivot = inverse_general_matrix(
                result.data(), response.data(), dimension);
            if (!(pivot > T(0))) {
                throw std::runtime_error(
                    "singular auxiliary response in Fourier block (m="
                    +std::to_string(projector.m())+",l="
                    +std::to_string(projector.l())+")");
            }
            return result;
        }

        std::vector<T> inverse_cost(trace_gram.size());
        const T cost_pivot = inverse_general_matrix(
            inverse_cost.data(), trace_gram.data(),
            continuation_dimension);
        if (!(cost_pivot > T(0))) {
            throw std::runtime_error(
                "singular continuation trace cost in Fourier block (m="
                +std::to_string(projector.m())+",l="
                +std::to_string(projector.l())
                +"); set extended:response_cost_ridge");
        }

        // X=H^{-1}M^T, S=M X, K=X S^{-1}.  Then M K=I and Kb is the
        // minimum-H-norm continuation among all solutions of Ma=b.
        std::vector<T> x(
            static_cast<std::size_t>(continuation_dimension)*dimension,
            T(0));
        for (int row = 0; row < continuation_dimension; ++row) {
            for (int column = 0; column < dimension; ++column) {
                for (int inner = 0;
                     inner < continuation_dimension; ++inner) {
                    x[static_cast<std::size_t>(row)*dimension+column] +=
                        inverse_cost[static_cast<std::size_t>(row)
                                     *continuation_dimension+inner]
                        *response[static_cast<std::size_t>(column)
                                  *continuation_dimension+inner];
                }
            }
        }
        std::vector<T> schur(
            static_cast<std::size_t>(dimension)*dimension, T(0));
        for (int row = 0; row < dimension; ++row) {
            for (int column = 0; column < dimension; ++column) {
                for (int inner = 0;
                     inner < continuation_dimension; ++inner) {
                    schur[static_cast<std::size_t>(row)*dimension+column] +=
                        response[static_cast<std::size_t>(row)
                                 *continuation_dimension+inner]
                        *x[static_cast<std::size_t>(inner)*dimension+column];
                }
            }
        }
        std::vector<T> inverse_schur(schur.size());
        const T response_pivot = inverse_general_matrix(
            inverse_schur.data(), schur.data(), dimension);
        if (!(response_pivot > T(0))) {
            throw std::runtime_error(
                "rank-deficient expanded auxiliary response in Fourier "
                "block (m="+std::to_string(projector.m())+",l="
                +std::to_string(projector.l())+")");
        }
        std::vector<T> result(x.size(), T(0));
        for (int row = 0; row < continuation_dimension; ++row) {
            for (int column = 0; column < dimension; ++column) {
                for (int inner = 0; inner < dimension; ++inner) {
                    result[static_cast<std::size_t>(row)*dimension+column] +=
                        x[static_cast<std::size_t>(row)*dimension+inner]
                        *inverse_schur[
                            static_cast<std::size_t>(inner)*dimension+column];
                }
            }
        }
        return result;
    }

    void validate_response_condition(
        const BlockProjector& projector, double condition) const {
        if (!std::isfinite(condition)
            || condition > response_condition_limit_) {
            throw std::runtime_error(
                "ill-conditioned auxiliary response in Fourier block (m="
                +std::to_string(projector.m())+",l="
                +std::to_string(projector.l())+"): condition="
                +std::to_string(condition));
        }
    }

    CorrectionBlock build_correction(const Config& stokes_config,
                                     const BlockProjector& projector) {
        NSCylFourierBlockNative<T> stokes(
            stokes_config, projector.m(), projector.l(), 1);
        const auto local_indices = local_velocity_indices(
            projector.phase_count());
        const auto full_auxiliary_indices =
            full_auxiliary_velocity_indices(projector.phase_count());
        const int velocity_size = static_cast<int>(local_indices.size());
        const int dimension = projector.dimension();
        const int continuation_dimension =
            dimension*response_basis_count_;
        if (velocity_size != stokes.velocity_block_size()) {
            throw std::logic_error(
                "auxiliary velocity layout does not match native block");
        }

        std::vector<T> matrix(
            static_cast<std::size_t>(velocity_size)*velocity_size);
        std::vector<T> input(stokes.size(), T(0));
        std::vector<T> output(stokes.size(), T(0));
        for (int column = 0; column < velocity_size; ++column) {
            std::fill(input.begin(), input.end(), T(0));
            input[local_indices[column]] = T(1);
            stokes.apply(output.data(), input.data());
            for (int row = 0; row < velocity_size; ++row) {
                matrix[static_cast<std::size_t>(column)*velocity_size+row] =
                    ((row == column ? T(1) : T(0))
                     -output[local_indices[row]])/static_cast<T>(geometry_.dt);
            }
        }

        const int constraint_count = stokes.pressure_block_size()
            -(stokes.m() == 0 && stokes.l() == 0 ? 1 : 0);
        const auto divergence = stokes.velocity_divergence_matrix(true);
        const int saddle_size = velocity_size+constraint_count;
        std::vector<T> saddle(
            static_cast<std::size_t>(saddle_size)*saddle_size, T(0));
        for (int column = 0; column < velocity_size; ++column) {
            for (int row = 0; row < velocity_size; ++row) {
                saddle[static_cast<std::size_t>(column)*saddle_size+row] =
                    matrix[static_cast<std::size_t>(column)*velocity_size+row];
            }
        }
        for (int pressure = 0; pressure < constraint_count; ++pressure) {
            for (int velocity = 0; velocity < velocity_size; ++velocity) {
                const T entry = divergence[
                    static_cast<std::size_t>(pressure)*velocity_size+velocity];
                saddle[static_cast<std::size_t>(velocity)*saddle_size
                       +velocity_size+pressure] = entry;
                saddle[static_cast<std::size_t>(velocity_size+pressure)
                       *saddle_size+velocity] = entry;
            }
        }

        std::vector<T> right_hand_sides(
            static_cast<std::size_t>(saddle_size)*continuation_dimension,
            T(0));
        for (int degree = 0; degree < response_basis_count_; ++degree) {
            const auto weights = local_velocity_weights(
                projector.phase_count(), degree);
            for (int mode = 0; mode < dimension; ++mode) {
                const int column = degree*dimension+mode;
                const auto& right = projector.right_basis()[mode];
                for (int row = 0; row < velocity_size; ++row) {
                    right_hand_sides[
                        static_cast<std::size_t>(column)*saddle_size+row] =
                        right[full_auxiliary_indices[row]]*weights[row];
                }
            }
        }
        solve_dense(saddle, right_hand_sides, saddle_size,
                    continuation_dimension);

        CorrectionBlock result;
        result.basis.assign(
            continuation_dimension,
            std::vector<T>(projector.block_size(), T(0)));
        for (int column = 0; column < continuation_dimension; ++column) {
            for (int row = 0; row < velocity_size; ++row) {
                result.basis[column][full_auxiliary_indices[row]] =
                    right_hand_sides[
                        static_cast<std::size_t>(column)*saddle_size+row];
            }
        }

        std::vector<T> response(
            static_cast<std::size_t>(dimension)*continuation_dimension,
            T(0));
        std::vector<T> response_column(dimension);
        for (int column = 0; column < continuation_dimension; ++column) {
            projector.coordinates(
                response_column.data(), result.basis[column].data());
            for (int row = 0; row < dimension; ++row) {
                response[static_cast<std::size_t>(row)
                         *continuation_dimension+column] =
                    response_column[row];
            }
        }
        result.response = response;
        result.trace_gram = build_trace_gram(projector, result.basis);
        result.inverse_response = minimum_trace_right_inverse(
            projector, result.response, result.trace_gram,
            dimension, continuation_dimension);
        result.response_norm = infinity_norm(
            response, dimension, continuation_dimension);
        result.inverse_response_norm = infinity_norm(
            result.inverse_response, continuation_dimension, dimension);
        result.response_condition = result.response_norm
            *result.inverse_response_norm;
        validate_response_condition(projector, result.response_condition);
        return result;
    }

    void build_corrections() {
        const Config stokes_config = auxiliary_config(geometry_.Re);
        corrections_.reserve(projector_.blocks().size());
        for (const auto& block : projector_.blocks()) {
            corrections_.push_back(build_correction(
                stokes_config, block));
        }
    }

    void compute_amplitudes(
        const CorrectionBlock& correction, const std::vector<T>& coordinates,
        std::vector<T>& amplitudes) const {
        const int dimension = static_cast<int>(coordinates.size());
        const int continuation_dimension =
            static_cast<int>(correction.basis.size());
        if (static_cast<int>(amplitudes.size()) != continuation_dimension) {
            throw std::invalid_argument(
                "auxiliary amplitude vector has the wrong size");
        }
        if (response_regularization_ == 0) {
            for (int row = 0; row < continuation_dimension; ++row) {
                for (int column = 0; column < dimension; ++column) {
                    amplitudes[row] -= correction.inverse_response[
                        static_cast<std::size_t>(row)*dimension+column]
                        *coordinates[column];
                }
            }
            return;
        }

        std::vector<T> normal(
            static_cast<std::size_t>(continuation_dimension)
                *continuation_dimension,
            T(0));
        std::vector<T> right_hand_side(continuation_dimension, T(0));
        for (int row = 0; row < continuation_dimension; ++row) {
            for (int coordinate = 0;
                 coordinate < dimension; ++coordinate) {
                right_hand_side[row] -= correction.response[
                    static_cast<std::size_t>(coordinate)
                        *continuation_dimension+row]
                    *coordinates[coordinate];
            }
            for (int column = 0;
                 column < continuation_dimension; ++column) {
                for (int coordinate = 0;
                     coordinate < dimension; ++coordinate) {
                    normal[static_cast<std::size_t>(row)
                           *continuation_dimension+column] +=
                        correction.response[
                            static_cast<std::size_t>(coordinate)
                                *continuation_dimension+row]
                        *correction.response[
                            static_cast<std::size_t>(coordinate)
                                *continuation_dimension+column];
                }
                normal[static_cast<std::size_t>(row)
                       *continuation_dimension+column] +=
                    static_cast<T>(response_regularization_)
                    *correction.trace_gram[
                        static_cast<std::size_t>(row)
                            *continuation_dimension+column];
            }
        }

        std::vector<T> inverse(normal.size());
        const T pivot = inverse_general_matrix(
            inverse.data(), normal.data(), continuation_dimension);
        if (!(pivot > T(0))) {
            throw std::runtime_error(
                "regularized auxiliary response solve failed");
        }
        for (int row = 0; row < continuation_dimension; ++row) {
            for (int column = 0;
                 column < continuation_dimension; ++column) {
                amplitudes[row] += inverse[
                    static_cast<std::size_t>(row)
                        *continuation_dimension+column]
                    *right_hand_side[column];
            }
        }
    }

    BlockPhysicalMetrics physical_block_metrics(
        const BlockProjector& projector,
        const std::vector<T>& reduced_block) {
        if (static_cast<int>(reduced_block.size()) != projector.block_size()) {
            throw std::invalid_argument(
                "block correction has the wrong packed size");
        }

        std::vector<T> expanded;
        const T* block = reduced_block.data();
        if (projector.pressure_gauge_fixed()) {
            expanded.resize(
                static_cast<std::size_t>(projector.phase_count())
                *layout_.radial_size);
            layout_.expand_zero_gauge_block(
                geometry_, reduced_block.data(), expanded.data());
            block = expanded.data();
        }

        std::vector<T> fourier(layout_.state_size, T(0));
        const auto phi = packed_indices(projector.m(), layout_.nphi);
        const auto z = packed_indices(projector.l(), layout_.nz);
        int phase = 0;
        for (int i : phi) {
            for (int k : z) {
                for (int j = 1; j < layout_.nr; ++j) {
                    fourier[state_index(Component::u, i, k, j)] = block[
                        static_cast<std::size_t>(phase)*layout_.radial_size
                        +layout_.radial_index(Component::u, j)];
                }
                for (Component component : {Component::v, Component::w}) {
                    for (int j = 1; j <= layout_.nr; ++j) {
                        fourier[state_index(component, i, k, j)] = block[
                            static_cast<std::size_t>(phase)
                                *layout_.radial_size
                            +layout_.radial_index(component, j)];
                    }
                }
                ++phase;
            }
        }

        std::vector<T> physical(layout_.state_size, T(0));
        std::vector<T> plane_fourier(fft_.size(), T(0));
        std::vector<T> plane_physical(fft_.size(), T(0));
        for (Component component : {
                 Component::u, Component::v, Component::w}) {
            const int radial_end = component == Component::u
                ? layout_.nr-1 : layout_.nr;
            for (int j = 1; j <= radial_end; ++j) {
                for (int i = 0; i < layout_.nphi; ++i) {
                    for (int k = 0; k < layout_.nz; ++k) {
                        plane_fourier[plane_index(i, k)] =
                            fourier[state_index(component, i, k, j)];
                    }
                }
                fft_.synthesis(plane_fourier.data(), plane_physical.data());
                for (int i = 0; i < layout_.nphi; ++i) {
                    for (int k = 0; k < layout_.nz; ++k) {
                        physical[state_index(component, i, k, j)] =
                            plane_physical[plane_index(i, k)];
                    }
                }
            }
        }

        BlockPhysicalMetrics result;
        result.velocity_norm = layout_.velocity_norm(
            geometry_, physical.data());
        long double boundary_sum = 0;
        for (int i = 0; i < layout_.nphi; ++i) {
            for (int k = 0; k < layout_.nz; ++k) {
                const long double radial = physical[state_index(
                    Component::u, i, k, original_nr_)];
                const long double axial = T(0.5)*(
                    physical[state_index(
                        Component::v, i, k, original_nr_)]
                    +physical[state_index(
                        Component::v, i, k, original_nr_+1)]);
                const long double azimuthal = T(0.5)*(
                    physical[state_index(
                        Component::w, i, k, original_nr_)]
                    +physical[state_index(
                        Component::w, i, k, original_nr_+1)]);
                const double magnitude2 = static_cast<double>(
                    radial*radial+axial*axial+azimuthal*azimuthal);
                boundary_sum += magnitude2;
                result.boundary_maximum = std::max(
                    result.boundary_maximum, std::sqrt(magnitude2));
            }
        }
        result.boundary_rms = std::sqrt(static_cast<double>(
            boundary_sum/(layout_.nphi*layout_.nz)));
        return result;
    }

    static double infinity_norm(const std::vector<T>& matrix,
                                int rows, int columns) {
        double result = 0;
        for (int row = 0; row < rows; ++row) {
            double sum = 0;
            for (int column = 0; column < columns; ++column) {
                sum += std::abs(static_cast<double>(matrix[
                    static_cast<std::size_t>(row)*columns+column]));
            }
            result = std::max(result, sum);
        }
        return result;
    }

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
        throw std::logic_error("unknown NSCyl component");
    }

    int state_index(Component component, int i, int k, int j) const {
        return component_offset(component)
            +(i*layout_.nz+k)*component_size(component)+(j-1);
    }

    static int packed_index(const Layout& layout, Component component,
                            int i, int k, int j) {
        const int size = component == Component::u ? layout.nr-1 : layout.nr;
        int offset = 0;
        switch (component) {
        case Component::u: offset = layout.u_offset; break;
        case Component::v: offset = layout.v_offset; break;
        case Component::w: offset = layout.w_offset; break;
        case Component::p: offset = layout.p_offset; break;
        }
        return offset+(i*layout.nz+k)*size+(j-1);
    }

    std::size_t plane_index(int i, int k) const {
        return static_cast<std::size_t>(i)*layout_.nz+k;
    }

    void analysis() {
        layout_.for_each_radial(
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
        layout_.for_each_radial(
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

    void canonicalize_pressure_gauge() {
        long double weighted_sum = 0;
        long double weight = 0;
        for (int j = 1; j <= layout_.nr; ++j) {
            const long double radius = geometry_.r0+(j-0.5L)*geometry_.dr;
            weighted_sum += radius*static_cast<long double>(packed_fourier_[
                state_index(Component::p, 0, 0, j)]);
            weight += radius;
        }
        const T mean = static_cast<T>(weighted_sum/weight);
        for (int j = 1; j <= layout_.nr; ++j) {
            packed_fourier_[state_index(Component::p, 0, 0, j)] -= mean;
        }
    }

    void gather_block(const BlockProjector& projector) {
        const auto phi = packed_indices(projector.m(), layout_.nphi);
        const auto z = packed_indices(projector.l(), layout_.nz);
        full_block_.assign(
            static_cast<std::size_t>(projector.phase_count())
                *layout_.radial_size,
            T(0));
        int phase = 0;
        for (int i : phi) {
            for (int k : z) {
                layout_.for_each_radial(
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
                geometry_, full_block_.data(), block_.data());
        } else {
            std::copy(full_block_.begin(), full_block_.end(), block_.begin());
        }
    }

    void scatter_block(const BlockProjector& projector, const T* block) {
        const T* full = block;
        if (projector.pressure_gauge_fixed()) {
            layout_.expand_zero_gauge_block(
                geometry_, block, full_block_.data());
            full = full_block_.data();
        }
        const auto phi = packed_indices(projector.m(), layout_.nphi);
        const auto z = packed_indices(projector.l(), layout_.nz);
        int phase = 0;
        for (int i : phi) {
            for (int k : z) {
                layout_.for_each_radial(
                    [&](Component component, int j, int radial_index) {
                        packed_fourier_[state_index(component, i, k, j)] =
                            full[static_cast<std::size_t>(phase)
                                     *layout_.radial_size+radial_index];
                    });
                ++phase;
            }
        }
    }

    double original_domain_velocity_norm(const T* state) const {
        long double sum = 0;
        const long double cell_measure = geometry_.dr
            *geometry_.dphi*geometry_.dz;
        for (int i = 0; i < geometry_.nphi; ++i) {
            for (int k = 0; k < geometry_.nz; ++k) {
                for (int j = 1; j < original_nr_; ++j) {
                    const int index = state_index(Component::u, i, k, j);
                    const long double radius = geometry_.r0+j*geometry_.dr;
                    sum += radius*static_cast<long double>(state[index])
                        *state[index];
                }
                for (Component component : {Component::v, Component::w}) {
                    for (int j = 1; j <= original_nr_; ++j) {
                        const int index = state_index(component, i, k, j);
                        const long double radius =
                            geometry_.r0+(j-0.5L)*geometry_.dr;
                        sum += radius*static_cast<long double>(state[index])
                            *state[index];
                    }
                }
            }
        }
        return std::sqrt(std::max(0.0,
            static_cast<double>(cell_measure*sum)));
    }
};

} // namespace fdm
