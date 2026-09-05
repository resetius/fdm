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
    double response_norm = 0;
    double inverse_response_norm = 0;
    double response_condition = 0;
    double unstable_coordinate_norm_before = 0;
    double unstable_coordinate_norm_after = 0;
    double coefficient_norm = 0;
};

struct NSCylExtendedFilterDiagnostics {
    double unstable_coordinate_norm_before = 0;
    double unstable_coordinate_norm_after = 0;
    double correction_velocity_norm = 0;
    double original_domain_change_norm = 0;
    std::vector<NSCylExtendedBlockFilterDiagnostics> blocks;
};

// Builds the continuation correction only in the auxiliary annulus. For each
// unstable real Fourier block, restricted right modes are passed through a
// discrete stationary Stokes inverse in omega. If
// C=(L^T R)^{-1}L^T are the biorthogonal unstable-coordinate functionals, the
// small system C*W*c=-C*q defines the support-restricted oblique projector
// W(CW)^{-1}C. Every velocity degree of freedom in the original cylinder is
// unchanged.
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
        if (!(response_condition_limit_ >= 1)) {
            throw std::invalid_argument(
                "extended response condition limit must be at least one");
        }
        validate_projector();
        build_corrections();
    }

    const Geometry& geometry() const { return geometry_; }
    int original_nr() const { return original_nr_; }
    int auxiliary_nr() const { return auxiliary_nr_; }
    double base_outer_radius() const { return base_outer_radius_; }

    NSCylExtendedFilterDiagnostics apply(
        std::vector<T>& extended_perturbation) {
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
            std::vector<T> amplitudes(dimension, T(0));
            projector.coordinates(before.data(), block_.data());
            for (int row = 0; row < dimension; ++row) {
                for (int column = 0; column < dimension; ++column) {
                    amplitudes[row] -= correction.inverse_response[
                        static_cast<std::size_t>(row)*dimension+column]
                        *before[column];
                }
            }
            for (int column = 0; column < dimension; ++column) {
                const auto& basis = correction.basis[column];
                for (int coordinate = 0;
                     coordinate < projector.block_size(); ++coordinate) {
                    block_[coordinate] += amplitudes[column]*basis[coordinate];
                }
            }
            projector.coordinates(after.data(), block_.data());
            scatter_block(projector, block_.data());

            NSCylExtendedBlockFilterDiagnostics block_result;
            block_result.m = projector.m();
            block_result.l = projector.l();
            block_result.response_norm = correction.response_norm;
            block_result.inverse_response_norm =
                correction.inverse_response_norm;
            block_result.response_condition = correction.response_condition;
            for (int coordinate = 0; coordinate < dimension; ++coordinate) {
                const long double b = before[coordinate];
                const long double a = after[coordinate];
                const long double c = amplitudes[coordinate];
                block_result.unstable_coordinate_norm_before +=
                    static_cast<double>(b*b);
                block_result.unstable_coordinate_norm_after +=
                    static_cast<double>(a*a);
                block_result.coefficient_norm += static_cast<double>(c*c);
            }
            block_result.unstable_coordinate_norm_before = std::sqrt(
                block_result.unstable_coordinate_norm_before);
            block_result.unstable_coordinate_norm_after = std::sqrt(
                block_result.unstable_coordinate_norm_after);
            block_result.coefficient_norm = std::sqrt(
                block_result.coefficient_norm);
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
    struct CorrectionBlock {
        std::vector<std::vector<T>> basis;
        std::vector<T> inverse_response;
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

    std::vector<T> phase_shift(const BlockProjector& projector,
                               bool azimuthal, int direction) const {
        const int phi_phases =
            (projector.m() == 0 || 2*projector.m() == geometry_.nphi)
            ? 1 : 2;
        const int z_phases =
            (projector.l() == 0 || 2*projector.l() == geometry_.nz)
            ? 1 : 2;
        const int one_count = azimuthal ? phi_phases : z_phases;
        const int frequency = azimuthal ? projector.m() : projector.l();
        const int grid_size = azimuthal ? geometry_.nphi : geometry_.nz;
        const T angle = static_cast<T>(
            direction*2*M_PI*static_cast<double>(frequency)/grid_size);
        std::vector<T> one(
            static_cast<std::size_t>(one_count)*one_count, T(0));
        if (one_count == 1) {
            one[0] = std::cos(angle);
        } else {
            const T cosine = std::cos(angle);
            const T sine = std::sin(angle);
            one[0] = cosine;
            one[1] = sine;
            one[2] = -sine;
            one[3] = cosine;
        }

        const int phases = projector.phase_count();
        std::vector<T> result(
            static_cast<std::size_t>(phases)*phases, T(0));
        for (int phi_row = 0; phi_row < phi_phases; ++phi_row) {
            for (int z_row = 0; z_row < z_phases; ++z_row) {
                const int row = phi_row*z_phases+z_row;
                for (int phi_column = 0;
                     phi_column < phi_phases; ++phi_column) {
                    for (int z_column = 0;
                         z_column < z_phases; ++z_column) {
                        const int column = phi_column*z_phases+z_column;
                        if (azimuthal && z_row == z_column) {
                            result[row*phases+column] =
                                one[phi_row*phi_phases+phi_column];
                        } else if (!azimuthal
                                   && phi_row == phi_column) {
                            result[row*phases+column] =
                                one[z_row*z_phases+z_column];
                        }
                    }
                }
            }
        }
        return result;
    }

    // Row-major matrix of the exact staggered divergence in omega.  The
    // normal velocity at both radial boundaries is zero and is therefore not
    // an unknown.  For the constant Fourier block one redundant conservation
    // row is omitted together with the pressure gauge.
    std::vector<T> divergence_matrix(
        const BlockProjector& projector,
        const std::vector<int>& velocity_indices,
        int& constraint_count) const {
        const Layout auxiliary(auxiliary_nr_, geometry_.nz, geometry_.nphi);
        const int phases = projector.phase_count();
        const bool constant = projector.m() == 0 && projector.l() == 0;
        constraint_count = phases*auxiliary_nr_-(constant ? 1 : 0);
        const int velocity_size = static_cast<int>(velocity_indices.size());
        std::vector<int> position(
            static_cast<std::size_t>(phases)*auxiliary.radial_size, -1);
        for (int index = 0; index < velocity_size; ++index) {
            position[velocity_indices[index]] = index;
        }
        std::vector<T> result(
            static_cast<std::size_t>(constraint_count)*velocity_size, T(0));
        const auto phi_minus = phase_shift(projector, true, -1);
        const auto z_minus = phase_shift(projector, false, -1);
        int row = 0;
        for (int phase = 0; phase < phases; ++phase) {
            for (int j = 1; j <= auxiliary_nr_; ++j) {
                if (constant && phase == phases-1 && j == auxiliary_nr_) {
                    continue;
                }
                const double radius = base_outer_radius_
                    +(j-0.5)*geometry_.dr;
                if (j < auxiliary_nr_) {
                    const int coordinate = phase*auxiliary.radial_size
                        +auxiliary.radial_index(Component::u, j);
                    result[static_cast<std::size_t>(row)*velocity_size
                           +position[coordinate]] += static_cast<T>(
                        (radius+0.5*geometry_.dr)
                        /(radius*geometry_.dr));
                }
                if (j > 1) {
                    const int coordinate = phase*auxiliary.radial_size
                        +auxiliary.radial_index(Component::u, j-1);
                    result[static_cast<std::size_t>(row)*velocity_size
                           +position[coordinate]] -= static_cast<T>(
                        (radius-0.5*geometry_.dr)
                        /(radius*geometry_.dr));
                }
                for (int column_phase = 0;
                     column_phase < phases; ++column_phase) {
                    const int v_coordinate =
                        column_phase*auxiliary.radial_size
                        +auxiliary.radial_index(Component::v, j);
                    const int w_coordinate =
                        column_phase*auxiliary.radial_size
                        +auxiliary.radial_index(Component::w, j);
                    const T identity = phase == column_phase ? T(1) : T(0);
                    result[static_cast<std::size_t>(row)*velocity_size
                           +position[v_coordinate]] +=
                        (identity-z_minus[phase*phases+column_phase])
                        /static_cast<T>(geometry_.dz);
                    result[static_cast<std::size_t>(row)*velocity_size
                           +position[w_coordinate]] +=
                        (identity-phi_minus[phase*phases+column_phase])
                        /static_cast<T>(radius*geometry_.dphi);
                }
                ++row;
            }
        }
        if (row != constraint_count) {
            throw std::logic_error("invalid auxiliary divergence size");
        }
        return result;
    }

    CorrectionBlock build_correction(const Config& stokes_config,
                                     const BlockProjector& projector) const {
        NSCylFourierBlockNative<T> stokes(
            stokes_config, projector.m(), projector.l(), 1);
        const auto local_indices = local_velocity_indices(
            projector.phase_count());
        const auto full_auxiliary_indices =
            full_auxiliary_velocity_indices(projector.phase_count());
        const int velocity_size = static_cast<int>(local_indices.size());
        const int dimension = projector.dimension();

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

        int constraint_count = 0;
        const auto divergence = divergence_matrix(
            projector, local_indices, constraint_count);
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
            static_cast<std::size_t>(saddle_size)*dimension, T(0));
        for (int column = 0; column < dimension; ++column) {
            const auto& right = projector.right_basis()[column];
            for (int row = 0; row < velocity_size; ++row) {
                right_hand_sides[
                    static_cast<std::size_t>(column)*saddle_size+row] =
                    right[full_auxiliary_indices[row]];
            }
        }
        solve_dense(saddle, right_hand_sides, saddle_size, dimension);

        CorrectionBlock result;
        result.basis.assign(
            dimension, std::vector<T>(projector.block_size(), T(0)));
        for (int column = 0; column < dimension; ++column) {
            for (int row = 0; row < velocity_size; ++row) {
                result.basis[column][full_auxiliary_indices[row]] =
                    right_hand_sides[
                        static_cast<std::size_t>(column)*saddle_size+row];
            }
        }

        std::vector<T> response(
            static_cast<std::size_t>(dimension)*dimension, T(0));
        std::vector<T> response_column(dimension);
        for (int column = 0; column < dimension; ++column) {
            projector.coordinates(
                response_column.data(), result.basis[column].data());
            for (int row = 0; row < dimension; ++row) {
                response[static_cast<std::size_t>(row)*dimension+column] =
                    response_column[row];
            }
        }
        result.inverse_response.resize(response.size());
        const T pivot = inverse_general_matrix(
            result.inverse_response.data(), response.data(), dimension);
        if (!(pivot > T(0))) {
            throw std::runtime_error(
                "singular auxiliary response in Fourier block (m="
                +std::to_string(projector.m())+",l="
                +std::to_string(projector.l())+")");
        }
        result.response_norm = infinity_norm(response, dimension);
        result.inverse_response_norm = infinity_norm(
            result.inverse_response, dimension);
        result.response_condition =
            result.response_norm*result.inverse_response_norm;
        if (!std::isfinite(result.response_condition)
            || result.response_condition > response_condition_limit_) {
            throw std::runtime_error(
                "ill-conditioned auxiliary response in Fourier block (m="
                +std::to_string(projector.m())+",l="
                +std::to_string(projector.l())+"): condition="
                +std::to_string(result.response_condition));
        }
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

    static double infinity_norm(const std::vector<T>& matrix, int size) {
        double result = 0;
        for (int row = 0; row < size; ++row) {
            double sum = 0;
            for (int column = 0; column < size; ++column) {
                sum += std::abs(static_cast<double>(matrix[
                    static_cast<std::size_t>(row)*size+column]));
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
