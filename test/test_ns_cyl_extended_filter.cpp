#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "ns_cyl.h"
#include "ns_cyl_checkpoint_storage.h"
#include "ns_cyl_extended_filter.h"
#include "ns_cyl_nonlinear_gluing.h"
#include "ns_cyl_spectral_filter.h"
#include "ns_cyl_spectral_storage.h"
#include "ns_cyl_state.h"

namespace {

using T = double;
using Task = fdm::NSCyl<T, true, fdm::tensor_flag::periodic>;
using Layout = fdm::NSCylStateLayout<T>;
using Component = Layout::Component;
using ExtendedFilter = fdm::NSCylExtendedSpectralFilter<T>;

// The zero-extended Couette field is a piecewise base state, not a stationary
// solution of an ordinary full-domain stencil across the Omega/omega
// interface.  Advance the exact nonlinear perturbation map instead.
class ExtendedPerturbationStepper {
public:
    ExtendedPerturbationStepper(
        const Config& config, std::vector<T> reference)
        : state_(config)
        , layout_(state_)
        , reference_(std::move(reference))
        , base_image_(layout_.state_size)
        , total_(layout_.state_size)
        , image_(layout_.state_size) {
        Task base(config);
        layout_.unpack(base, reference_.data());
        base.step();
        layout_.pack(base, base_image_.data());
    }

    void step(std::vector<T>& perturbation) {
        if (static_cast<int>(perturbation.size()) != layout_.state_size) {
            throw std::invalid_argument(
                "extended perturbation step has the wrong state size");
        }
        for (int index = 0; index < layout_.state_size; ++index) {
            total_[index] = reference_[index]+perturbation[index];
        }
        layout_.unpack(state_, total_.data());
        state_.step();
        layout_.pack(state_, image_.data());
        for (int index = 0; index < layout_.state_size; ++index) {
            perturbation[index] = image_[index]-base_image_[index];
        }
        layout_.normalize_packed_pressure(state_, perturbation.data());
    }

private:
    Task state_;
    Layout layout_;
    std::vector<T> reference_;
    std::vector<T> base_image_;
    std::vector<T> total_;
    std::vector<T> image_;
};

std::string number(double value) {
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

bool nearly_equal(double first, double second) {
    const double scale = std::max({1.0, std::abs(first), std::abs(second)});
    return std::abs(first-second)
        <= 128*std::numeric_limits<double>::epsilon()*scale;
}

Config make_extended_config(
    const fdm::NSCylSpectralMetadata& metadata,
    double response_condition_limit, double response_regularization,
    int response_basis_count, int response_cost_horizon_steps,
    int response_cost_sample_stride, double response_cost_ridge,
    const std::string& response_cost) {
    Config result;
    std::vector<std::string> arguments = {
        "fdm_ns_cyl_extended_filter",
        "--ns:r="+number(metadata.r),
        "--ns:R="+number(metadata.R),
        "--ns:h1="+number(metadata.h1),
        "--ns:h2="+number(metadata.h2),
        "--ns:u0="+number(metadata.wall_speed),
        "--ns:Re="+number(metadata.reynolds),
        "--ns:dt="+number(metadata.dt),
        "--ns:nr="+std::to_string(metadata.nr),
        "--ns:nphi="+std::to_string(metadata.nphi),
        "--ns:nz="+std::to_string(metadata.nz),
        "--ns:verbose=0",
        "--spectral:base_outer_radius="+number(metadata.base_outer_radius),
        "--extended:response_condition_limit="
            +number(response_condition_limit),
        "--extended:response_regularization="
            +number(response_regularization),
        "--extended:response_basis_count="
            +std::to_string(response_basis_count),
        "--extended:response_cost_horizon_steps="
            +std::to_string(response_cost_horizon_steps),
        "--extended:response_cost_sample_stride="
            +std::to_string(response_cost_sample_stride),
        "--extended:response_cost_ridge="+number(response_cost_ridge),
        "--extended:response_cost="+response_cost
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    result.rewrite(static_cast<int>(argv.size()), argv.data());
    return result;
}

void validate_domains(const Task& original,
                      const fdm::NSCylSpectralMetadata& extended) {
    if (extended.scalar_type != "float64"
        || extended.nphi != original.nphi || extended.nz != original.nz
        || extended.nr <= original.nr
        || !nearly_equal(extended.r, original.r0)
        || !nearly_equal(extended.base_outer_radius, original.R)
        || !nearly_equal(extended.h1, original.h1)
        || !nearly_equal(extended.h2, original.h2)
        || !nearly_equal(extended.reynolds, original.Re)
        || !nearly_equal(extended.dt, original.dt)
        || !nearly_equal(extended.wall_speed, original.U0)) {
        throw std::runtime_error(
            "original checkpoint and extended spectrum are incompatible");
    }
    const double extended_dr = (extended.R-extended.r)/extended.nr;
    if (!nearly_equal(extended_dr, original.dr)) {
        throw std::runtime_error(
            "Omega and G must use the same radial mesh spacing");
    }
}

fdm::NSCylSpectralModeSet<T> select_control_modes(
    const fdm::NSCylSpectralModeSet<T>& source, double minimum_growth) {
    fdm::NSCylSpectralModeSet<T> result;
    for (const auto& mode : source.modes()) {
        if (mode.growth_rate >= minimum_growth) {
            result.append_filterable_mode(mode);
        }
    }
    result.sort_by_block_and_growth();
    return result;
}

int state_index(const Layout& layout, Component component,
                int i, int k, int j) {
    int offset = 0;
    int radial_size = layout.nr;
    switch (component) {
    case Component::u:
        offset = layout.u_offset;
        radial_size = layout.nr-1;
        break;
    case Component::v:
        offset = layout.v_offset;
        break;
    case Component::w:
        offset = layout.w_offset;
        break;
    case Component::p:
        offset = layout.p_offset;
        break;
    }
    return offset+(i*layout.nz+k)*radial_size+(j-1);
}

std::vector<T> couette_reference(Task& task, const Layout& layout) {
    layout.initialize_couette_state(task);
    auto result = layout.pack(task);
    layout.normalize_packed_pressure(task, result.data());
    return result;
}

double maximum_divergence(
    const std::vector<T>& state, const ExtendedFilter::Geometry& geometry) {
    const Layout layout(geometry.nr, geometry.nz, geometry.nphi);
    double result = 0;
    for (int i = 0; i < geometry.nphi; ++i) {
        const int im = (i+geometry.nphi-1)%geometry.nphi;
        for (int k = 0; k < geometry.nz; ++k) {
            const int km = (k+geometry.nz-1)%geometry.nz;
            for (int j = 1; j <= geometry.nr; ++j) {
                const double radius = geometry.r0+(j-0.5)*geometry.dr;
                const double inner_radius = radius-0.5*geometry.dr;
                const double outer_radius = radius+0.5*geometry.dr;
                const double radial_outer = j < geometry.nr
                    ? state[state_index(layout, Component::u, i, k, j)] : 0;
                const double radial_inner = j > 1
                    ? state[state_index(layout, Component::u, i, k, j-1)] : 0;
                const double axial = (
                    state[state_index(layout, Component::v, i, k, j)]
                    -state[state_index(layout, Component::v, i, km, j)])
                    /geometry.dz;
                const double azimuthal = (
                    state[state_index(layout, Component::w, i, k, j)]
                    -state[state_index(layout, Component::w, im, k, j)])
                    /(radius*geometry.dphi);
                const double divergence =
                    (outer_radius*radial_outer
                     -inner_radius*radial_inner)/(radius*geometry.dr)
                    +axial+azimuthal;
                result = std::max(result, std::abs(divergence));
            }
        }
    }
    return result;
}

double original_domain_velocity_norm(
    const std::vector<T>& state, const ExtendedFilter::Geometry& geometry,
    int original_nr) {
    const Layout layout(geometry.nr, geometry.nz, geometry.nphi);
    std::vector<T> restricted(state.size(), T(0));
    for (int i = 0; i < geometry.nphi; ++i) {
        for (int k = 0; k < geometry.nz; ++k) {
            for (int j = 1; j < original_nr; ++j) {
                restricted[state_index(
                    layout, Component::u, i, k, j)] = state[state_index(
                        layout, Component::u, i, k, j)];
            }
            for (Component component : {Component::v, Component::w}) {
                for (int j = 1; j <= original_nr; ++j) {
                    restricted[state_index(
                        layout, component, i, k, j)] = state[state_index(
                            layout, component, i, k, j)];
                }
            }
        }
    }
    return layout.velocity_norm(geometry, restricted.data());
}

struct TraceNorm {
    double rms = 0;
    double maximum = 0;
};

TraceNorm boundary_trace_norm(
    const std::vector<T>& state, const ExtendedFilter::Geometry& geometry,
    int original_nr) {
    const Layout layout(geometry.nr, geometry.nz, geometry.nphi);
    long double sum = 0;
    TraceNorm result;
    for (int i = 0; i < geometry.nphi; ++i) {
        for (int k = 0; k < geometry.nz; ++k) {
            const double radial = state[state_index(
                layout, Component::u, i, k, original_nr)];
            const double axial = 0.5*(
                state[state_index(layout, Component::v, i, k, original_nr)]
                +state[state_index(
                    layout, Component::v, i, k, original_nr+1)]);
            const double azimuthal = 0.5*(
                state[state_index(layout, Component::w, i, k, original_nr)]
                +state[state_index(
                    layout, Component::w, i, k, original_nr+1)]);
            const double magnitude2 =
                radial*radial+axial*axial+azimuthal*azimuthal;
            sum += magnitude2;
            result.maximum = std::max(result.maximum, std::sqrt(magnitude2));
        }
    }
    result.rms = std::sqrt(static_cast<double>(
        sum/(geometry.nphi*geometry.nz)));
    return result;
}

fdm::NSCylOuterBoundaryVelocity<T> boundary_trace_difference(
    const std::vector<T>& controlled,
    const std::vector<T>& uncontrolled,
    const ExtendedFilter::Geometry& geometry, int original_nr) {
    if (controlled.size() != uncontrolled.size()) {
        throw std::invalid_argument(
            "extended trace states have different sizes");
    }
    const Layout layout(geometry.nr, geometry.nz, geometry.nphi);
    fdm::NSCylOuterBoundaryVelocity<T> result;
    result.nphi = geometry.nphi;
    result.nz = geometry.nz;
    const std::size_t plane_size =
        static_cast<std::size_t>(geometry.nphi)*geometry.nz;
    result.radial.resize(plane_size);
    result.axial.resize(plane_size);
    result.azimuthal.resize(plane_size);
    for (int i = 0; i < geometry.nphi; ++i) {
        for (int k = 0; k < geometry.nz; ++k) {
            const std::size_t plane =
                static_cast<std::size_t>(i)*geometry.nz+k;
            auto difference = [&](Component component, int j) {
                const int index = state_index(
                    layout, component, i, k, j);
                return controlled[index]-uncontrolled[index];
            };
            result.radial[plane] = difference(Component::u, original_nr);
            result.axial[plane] = T(0.5)*(
                difference(Component::v, original_nr)
                +difference(Component::v, original_nr+1));
            result.azimuthal[plane] = T(0.5)*(
                difference(Component::w, original_nr)
                +difference(Component::w, original_nr+1));
        }
    }
    return result;
}

void write_evolution_row(
    std::ofstream& output, const char* branch, int step, double time,
    bool filter_applied,
    const fdm::NSCylExtendedFilterDiagnostics& modal,
    const std::vector<T>& state, const ExtendedFilter::Geometry& geometry,
    int original_nr) {
    const Layout layout(geometry.nr, geometry.nz, geometry.nphi);
    const auto trace = boundary_trace_norm(state, geometry, original_nr);
    output << branch << ',' << step << ',' << time << ','
           << (filter_applied ? 1 : 0) << ','
           << modal.unstable_coordinate_norm_before << ','
           << modal.unstable_coordinate_norm_after << ','
           << layout.velocity_norm(geometry, state.data()) << ','
           << original_domain_velocity_norm(state, geometry, original_nr)
           << ',' << maximum_divergence(state, geometry) << ','
           << trace.rms << ',' << trace.maximum << ','
           << modal.correction_velocity_norm << '\n';
}

void write_diagnostics(
    const std::string& filename,
    const fdm::NSCylExtendedFilterDiagnostics& diagnostics) {
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("cannot create diagnostics CSV: "+filename);
    }
    output << "m,l,continuation_dimension,response_norm,"
              "inverse_response_norm,response_condition,"
              "coordinate_norm_before,"
              "coordinate_norm_after,coefficient_norm,"
              "correction_velocity_norm,boundary_rms,boundary_maximum,"
              "coordinate_to_correction_gain,"
              "coordinate_to_boundary_rms_gain\n";
    output << std::scientific << std::setprecision(16);
    for (const auto& block : diagnostics.blocks) {
        output << block.m << ',' << block.l << ','
               << block.continuation_dimension << ','
               << block.response_norm << ','
               << block.inverse_response_norm << ','
               << block.response_condition << ','
               << block.unstable_coordinate_norm_before << ','
               << block.unstable_coordinate_norm_after << ','
               << block.coefficient_norm << ','
               << block.correction_velocity_norm << ','
               << block.boundary_rms << ','
               << block.boundary_maximum << ','
               << block.coordinate_to_correction_gain << ','
               << block.coordinate_to_boundary_rms_gain << '\n';
    }
}

void write_trace(
    const std::string& filename,
    const fdm::NSCylOuterBoundaryVelocity<T>& boundary,
    const ExtendedFilter::Geometry& geometry) {
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("cannot create boundary trace CSV: "+filename);
    }
    output << "iphi,iz,phi,z,u_r,u_z,u_phi\n";
    output << std::scientific << std::setprecision(16);
    for (int i = 0; i < geometry.nphi; ++i) {
        for (int k = 0; k < geometry.nz; ++k) {
            const std::size_t index = static_cast<std::size_t>(i)*geometry.nz+k;
            output << i << ',' << k << ',' << i*geometry.dphi << ','
                   << geometry.h1+k*geometry.dz << ','
                   << boundary.radial[index] << ','
                   << boundary.axial[index] << ','
                   << boundary.azimuthal[index] << '\n';
        }
    }
}

std::vector<T> perturbation(Task& state, const Layout& layout,
                            const std::vector<T>& reference) {
    auto result = layout.pack(state);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] -= reference[index];
    }
    layout.normalize_packed_pressure(state, result.data());
    return result;
}

double maximum_divergence(Task& state) {
    double result = 0;
    for (int i = 0; i < state.nphi; ++i) {
        const int im = (i+state.nphi-1)%state.nphi;
        for (int k = 0; k < state.nz; ++k) {
            const int km = (k+state.nz-1)%state.nz;
            for (int j = 1; j <= state.nr; ++j) {
                const double radius = state.r0+(j-0.5)*state.dr;
                const double divergence =
                    ((radius+0.5*state.dr)*state.u[i][k][j]
                     -(radius-0.5*state.dr)*state.u[i][k][j-1])
                        /(radius*state.dr)
                    +(state.v[i][k][j]-state.v[i][km][j])/state.dz
                    +(state.w[i][k][j]-state.w[im][k][j])
                        /(radius*state.dphi);
                result = std::max(result, std::abs(divergence));
            }
        }
    }
    return result;
}

struct BoundaryEvolutionResult {
    std::vector<T> uncontrolled;
    std::vector<T> controlled;
};

class ExtendedNonlinearMethod {
public:
    using value_type = T;

    ExtendedNonlinearMethod(
        const Config& config, const std::vector<T>& reference,
        fdm::NSCylSpectralFilter<T>& filter,
        std::vector<std::vector<std::complex<T>>> multipliers,
        int map_steps, double power)
        : reference_(reference)
        , filter_(filter)
        , multipliers_(std::move(multipliers))
        , map_steps_(map_steps)
        , power_(power)
        , geometry_(config)
        , layout_(geometry_)
        , stepper_(config, reference) {
        if (map_steps_ <= 0 || !(power_ > 0)) {
            throw std::invalid_argument(
                "invalid extended nonlinear map parameters");
        }
    }

    std::vector<T> zero() const {
        return std::vector<T>(layout_.state_size, T(0));
    }

    std::vector<T> S(const std::vector<T>& perturbation) {
        std::vector<T> result = perturbation;
        for (int step = 0; step < map_steps_; ++step) {
            stepper_.step(result);
        }
        ++applications_;
        return result;
    }

    std::vector<T> Pminus(const std::vector<T>& perturbation) {
        auto state = make_state(perturbation);
        filter_.remove_packed(geometry_, state, reference_);
        return make_perturbation(state);
    }

    std::vector<T> Pplus(const std::vector<T>& perturbation) {
        auto stable = Pminus(perturbation);
        for (std::size_t index = 0; index < stable.size(); ++index) {
            stable[index] = perturbation[index]-stable[index];
        }
        return stable;
    }

    std::vector<T> PplusLinv(const std::vector<T>& perturbation) {
        auto state = make_state(perturbation);
        filter_.scale_unstable_packed(
            geometry_, state, reference_,
            [&](std::size_t block, std::vector<T>& coordinates) {
                divide(multipliers_.at(block), coordinates);
            });
        return make_perturbation(state);
    }

    double velocity_norm(const std::vector<T>& perturbation) const {
        return layout_.velocity_norm(geometry_, perturbation.data());
    }

    long long applications() const { return applications_; }

private:
    std::vector<T> make_state(const std::vector<T>& perturbation) const {
        if (static_cast<int>(perturbation.size()) != layout_.state_size) {
            throw std::invalid_argument(
                "extended nonlinear perturbation has the wrong size");
        }
        std::vector<T> result(perturbation.size());
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] = reference_[index]+perturbation[index];
        }
        return result;
    }

    std::vector<T> make_perturbation(const std::vector<T>& state) const {
        std::vector<T> result(state.size());
        for (std::size_t index = 0; index < result.size(); ++index) {
            result[index] = state[index]-reference_[index];
        }
        return result;
    }

    void divide(const std::vector<std::complex<T>>& multipliers,
                std::vector<T>& coordinates) const {
        for (std::size_t index = 0; index < coordinates.size();) {
            const std::complex<double> multiplier = std::pow(
                std::complex<double>(multipliers[index].real(),
                                     multipliers[index].imag()),
                power_);
            if (index+1 < coordinates.size()
                && multipliers[index].imag() != T(0)
                && multipliers[index+1] == multipliers[index]) {
                const std::complex<double> value(
                    coordinates[index], coordinates[index+1]);
                const auto scaled = value/std::conj(multiplier);
                coordinates[index] = static_cast<T>(scaled.real());
                coordinates[index+1] = static_cast<T>(scaled.imag());
                index += 2;
            } else {
                coordinates[index] = static_cast<T>(
                    coordinates[index]/multiplier.real());
                ++index;
            }
        }
    }

    const std::vector<T>& reference_;
    fdm::NSCylSpectralFilter<T>& filter_;
    std::vector<std::vector<std::complex<T>>> multipliers_;
    int map_steps_;
    double power_;
    Task geometry_;
    Layout layout_;
    ExtendedPerturbationStepper stepper_;
    long long applications_ = 0;
};

BoundaryEvolutionResult run_boundary_evolution(
    const Config& original_config, ExtendedFilter& filter,
    ExtendedNonlinearMethod* nonlinear_method, int nonlinear_iterations,
    const std::vector<T>& reference, const std::vector<T>& initial,
    int initial_time_index, int steps, int log_interval,
    int feedback_interval, double maximum_velocity_norm,
    const std::string& output_name) {
    if (steps < 0 || log_interval <= 0 || feedback_interval <= 0
        || !(maximum_velocity_norm > 0) || output_name.empty()) {
        throw std::invalid_argument("invalid boundary evolution settings");
    }

    Task uncontrolled(original_config);
    Task controlled(original_config);
    const Layout layout(uncontrolled);
    layout.unpack_sum(uncontrolled, reference, initial.data());
    layout.unpack_sum(controlled, reference, initial.data());

    std::ofstream output(output_name);
    if (!output) {
        throw std::runtime_error(
            "cannot create boundary evolution CSV: "+output_name);
    }
    output << "branch,step,time,feedback_applied,coordinate_norm,"
              "target_coordinate_norm,response_residual_norm,velocity_norm,"
              "maximum_divergence,boundary_rms,boundary_maximum,"
              "supported_correction_norm,gluing_correction_velocity_norm,"
              "nonlinear_S_applications\n";
    output << std::scientific << std::setprecision(16);

    fdm::NSCylOuterBoundaryVelocity<T> control;
    fdm::NSCylExtendedFilterDiagnostics controlled_modal;
    double target_coordinate_norm = 0;
    double response_residual_norm = 0;
    double supported_correction_norm = 0;
    double gluing_correction_velocity_norm = 0;
    std::unique_ptr<fdm::NSCylNonlinearGluing<ExtendedNonlinearMethod>> glue;
    if (nonlinear_method != nullptr) {
        if (nonlinear_iterations < 0) {
            throw std::invalid_argument(
                "boundary nonlinear iterations must be nonnegative");
        }
        glue = std::make_unique<
            fdm::NSCylNonlinearGluing<ExtendedNonlinearMethod>>(
                *nonlinear_method);
    }
    for (int step = 0; step <= steps; ++step) {
        const bool feedback = step%feedback_interval == 0;
        if (feedback) {
            auto q = perturbation(controlled, layout, reference);
            auto extended = filter.embed_original_perturbation(q);
            if (nonlinear_method == nullptr) {
                controlled_modal = filter.apply(extended);
                target_coordinate_norm = 0;
                response_residual_norm =
                    controlled_modal.unstable_coordinate_norm_after;
                gluing_correction_velocity_norm = 0;
            } else {
                auto diagnostic = extended;
                controlled_modal = filter.apply(diagnostic);
                const auto stable = nonlinear_method->Pminus(extended);
                glue->clear();
                const auto nonlinear = (*glue)(
                    stable, nonlinear_iterations);
                std::vector<T> target(stable.size());
                for (std::size_t index = 0; index < target.size(); ++index) {
                    target[index] = stable[index]+nonlinear[index];
                }
                auto target_diagnostic = target;
                target_coordinate_norm = filter.apply(
                    target_diagnostic).unstable_coordinate_norm_before;
                gluing_correction_velocity_norm =
                    nonlinear_method->velocity_norm(nonlinear);
                const auto response = filter.apply_towards(extended, target);
                response_residual_norm =
                    response.unstable_coordinate_norm_after;
                supported_correction_norm =
                    response.correction_velocity_norm;
            }
            if (nonlinear_method == nullptr) {
                supported_correction_norm =
                    controlled_modal.correction_velocity_norm;
            }
            control = filter.correction_boundary_velocity();
            controlled.set_outer_boundary_velocity(
                control.radial, control.axial, control.azimuthal);
            controlled.apply_boundary_conditions();
        }

        const bool log = step == 0 || step == steps
            || step%log_interval == 0 || feedback;
        if (log) {
            auto q_uncontrolled = perturbation(
                uncontrolled, layout, reference);
            auto extended_uncontrolled =
                filter.embed_original_perturbation(q_uncontrolled);
            const auto uncontrolled_modal = filter.apply(
                extended_uncontrolled);
            output << "uncontrolled," << step << ','
                   << (initial_time_index+step)*filter.geometry().dt
                   << ",0,"
                   << uncontrolled_modal.unstable_coordinate_norm_before
                   << ",0,0"
                   << ',' << layout.velocity_norm(
                       uncontrolled, q_uncontrolled.data())
                   << ',' << maximum_divergence(uncontrolled)
                   << ",0,0,0,0,0\n";

            auto q_controlled = perturbation(controlled, layout, reference);
            if (!feedback) {
                auto extended_controlled =
                    filter.embed_original_perturbation(q_controlled);
                controlled_modal = filter.apply(extended_controlled);
            }
            output << "boundary," << step << ','
                   << (initial_time_index+step)*filter.geometry().dt << ','
                   << (feedback ? 1 : 0) << ','
                   << controlled_modal.unstable_coordinate_norm_before << ','
                   << target_coordinate_norm << ','
                   << response_residual_norm << ','
                   << layout.velocity_norm(controlled, q_controlled.data())
                   << ',' << maximum_divergence(controlled) << ','
                   << control.rms_norm() << ',' << control.maximum_norm()
                   << ',' << supported_correction_norm << ','
                   << gluing_correction_velocity_norm << ','
                   << (nonlinear_method == nullptr
                           ? 0 : nonlinear_method->applications())
                   << '\n';

            const double unorm = layout.velocity_norm(
                uncontrolled, q_uncontrolled.data());
            const double cnorm = layout.velocity_norm(
                controlled, q_controlled.data());
            if (!std::isfinite(unorm) || !std::isfinite(cnorm)
                || unorm > maximum_velocity_norm
                || cnorm > maximum_velocity_norm) {
                throw std::runtime_error(
                    "boundary evolution exceeded the velocity norm limit at "
                    "step "+std::to_string(step));
            }
        }

        if (step != steps) {
            uncontrolled.step();
            controlled.step();
        }
    }

    return {layout.pack(uncontrolled), layout.pack(controlled)};
}

BoundaryEvolutionResult run_extended_trace_evolution(
    const Config& original_config, const Config& extended_config,
    ExtendedFilter& filter, ExtendedNonlinearMethod* nonlinear_method,
    int nonlinear_iterations, const std::vector<T>& extended_reference,
    const std::vector<T>& original_reference,
    const std::vector<T>& initial, int initial_time_index, int steps,
    int log_interval, int reorthogonalization_interval,
    double maximum_velocity_norm, const std::string& output_name) {
    if (steps < 0 || log_interval <= 0
        || reorthogonalization_interval <= 0
        || !(maximum_velocity_norm > 0) || output_name.empty()) {
        throw std::invalid_argument(
            "invalid extended trace evolution settings");
    }

    Task original_uncontrolled(original_config);
    Task original_controlled(original_config);
    const Layout original_layout(original_uncontrolled);
    original_layout.unpack_sum(
        original_uncontrolled, original_reference, initial.data());
    original_layout.unpack_sum(
        original_controlled, original_reference, initial.data());

    std::vector<T> extended_uncontrolled =
        filter.embed_original_perturbation(initial);
    std::vector<T> extended_controlled = extended_uncontrolled;
    ExtendedPerturbationStepper extended_stepper(
        extended_config, extended_reference);
    const Layout extended_layout(
        filter.geometry().nr, filter.geometry().nz,
        filter.geometry().nphi);

    std::ofstream output(output_name);
    if (!output) {
        throw std::runtime_error(
            "cannot create extended trace evolution CSV: "+output_name);
    }
    output << "branch,step,time,filter_applied,coordinate_norm,velocity_norm,"
              "maximum_divergence,boundary_rms,boundary_maximum,"
              "extended_coordinate_norm,extended_delta_velocity_norm,"
              "extended_delta_Omega_velocity_norm,target_coordinate_norm,"
              "response_residual_norm,gluing_correction_velocity_norm,"
              "nonlinear_S_applications\n";
    output << std::scientific << std::setprecision(16);

    std::unique_ptr<fdm::NSCylNonlinearGluing<ExtendedNonlinearMethod>> glue;
    if (nonlinear_method != nullptr) {
        if (nonlinear_iterations < 0) {
            throw std::invalid_argument(
                "extended trace nonlinear iterations must be nonnegative");
        }
        glue = std::make_unique<
            fdm::NSCylNonlinearGluing<ExtendedNonlinearMethod>>(
                *nonlinear_method);
    }

    double target_coordinate_norm = 0;
    double response_residual_norm = 0;
    double gluing_correction_velocity_norm = 0;
    for (int step = 0; step <= steps; ++step) {
        const bool apply = step%reorthogonalization_interval == 0;
        if (apply) {
            if (nonlinear_method == nullptr) {
                const auto response = filter.apply(extended_controlled);
                target_coordinate_norm = 0;
                response_residual_norm =
                    response.unstable_coordinate_norm_after;
                gluing_correction_velocity_norm = 0;
            } else {
                const auto stable =
                    nonlinear_method->Pminus(extended_controlled);
                glue->clear();
                const auto nonlinear = (*glue)(
                    stable, nonlinear_iterations);
                std::vector<T> target(stable.size());
                for (std::size_t index = 0; index < target.size(); ++index) {
                    target[index] = stable[index]+nonlinear[index];
                }
                auto target_diagnostic = target;
                target_coordinate_norm = filter.apply(
                    target_diagnostic).unstable_coordinate_norm_before;
                gluing_correction_velocity_norm =
                    nonlinear_method->velocity_norm(nonlinear);
                const auto response = filter.apply_towards(
                    extended_controlled, target);
                response_residual_norm =
                    response.unstable_coordinate_norm_after;
            }
        }

        const auto control = boundary_trace_difference(
            extended_controlled, extended_uncontrolled,
            filter.geometry(), filter.original_nr());
        original_controlled.set_outer_boundary_velocity(
            control.radial, control.axial, control.azimuthal);
        original_controlled.apply_boundary_conditions();

        const bool log = step == 0 || step == steps
            || step%log_interval == 0 || apply;
        if (log) {
            const auto q_uncontrolled = perturbation(
                original_uncontrolled, original_layout,
                original_reference);
            const auto q_controlled = perturbation(
                original_controlled, original_layout, original_reference);
            auto embedded_uncontrolled =
                filter.embed_original_perturbation(q_uncontrolled);
            auto embedded_controlled =
                filter.embed_original_perturbation(q_controlled);
            const auto uncontrolled_modal = filter.apply(
                embedded_uncontrolled);
            const auto controlled_modal = filter.apply(
                embedded_controlled);
            auto extended_diagnostic = extended_controlled;
            const auto extended_modal = filter.apply(
                extended_diagnostic);

            std::vector<T> extended_delta(extended_controlled.size());
            for (std::size_t index = 0;
                 index < extended_delta.size(); ++index) {
                extended_delta[index] = extended_controlled[index]
                    -extended_uncontrolled[index];
            }
            const double extended_delta_norm = extended_layout.velocity_norm(
                filter.geometry(), extended_delta.data());
            const double extended_delta_omega_norm =
                original_domain_velocity_norm(
                    extended_delta, filter.geometry(),
                    filter.original_nr());

            auto write = [&](const char* branch,
                             bool controlled_branch,
                             const fdm::NSCylExtendedFilterDiagnostics& modal,
                             const std::vector<T>& q, Task& task) {
                output << branch << ',' << step << ','
                       << (initial_time_index+step)*filter.geometry().dt
                       << ',' << (controlled_branch && apply ? 1 : 0) << ','
                       << modal.unstable_coordinate_norm_before << ','
                       << original_layout.velocity_norm(task, q.data())
                       << ',' << maximum_divergence(task) << ','
                       << control.rms_norm() << ','
                       << control.maximum_norm() << ','
                       << extended_modal.unstable_coordinate_norm_before
                       << ',' << extended_delta_norm << ','
                       << extended_delta_omega_norm << ','
                       << target_coordinate_norm << ','
                       << response_residual_norm << ','
                       << gluing_correction_velocity_norm << ','
                       << (nonlinear_method == nullptr
                               ? 0 : nonlinear_method->applications())
                       << '\n';
            };
            write("uncontrolled", false, uncontrolled_modal, q_uncontrolled,
                  original_uncontrolled);
            write("boundary", true, controlled_modal, q_controlled,
                  original_controlled);

            const double uncontrolled_norm = original_layout.velocity_norm(
                original_uncontrolled, q_uncontrolled.data());
            const double controlled_norm = original_layout.velocity_norm(
                original_controlled, q_controlled.data());
            if (!std::isfinite(uncontrolled_norm)
                || !std::isfinite(controlled_norm)
                || !std::isfinite(extended_delta_norm)
                || uncontrolled_norm > maximum_velocity_norm
                || controlled_norm > maximum_velocity_norm
                || extended_delta_norm > maximum_velocity_norm) {
                throw std::runtime_error(
                    "extended trace evolution exceeded the velocity norm "
                    "limit at step "+std::to_string(step));
            }
        }

        if (step != steps) {
            extended_stepper.step(extended_uncontrolled);
            extended_stepper.step(extended_controlled);
            const auto next_control = boundary_trace_difference(
                extended_controlled, extended_uncontrolled,
                filter.geometry(), filter.original_nr());

            original_uncontrolled.step();
            original_controlled.set_outer_boundary_step_data(
                control.radial, control.axial, control.azimuthal,
                next_control.radial, next_control.axial,
                next_control.azimuthal);
            original_controlled.step();
        }
    }

    return {original_layout.pack(original_uncontrolled),
            original_layout.pack(original_controlled)};
}

struct EvolutionResult {
    std::vector<T> uncontrolled;
    std::vector<T> controlled;
};

EvolutionResult run_evolution(
    const Config& extended_config, ExtendedFilter& filter,
    const std::vector<T>& reference, const std::vector<T>& initial,
    int initial_time_index, int steps, int log_interval,
    int reorthogonalization_interval, double maximum_velocity_norm,
    const std::string& output_name) {
    if (steps < 0 || log_interval <= 0
        || reorthogonalization_interval < 0
        || !(maximum_velocity_norm > 0) || output_name.empty()) {
        throw std::invalid_argument("invalid extended evolution settings");
    }
    std::ofstream output(output_name);
    if (!output) {
        throw std::runtime_error(
            "cannot create extended evolution CSV: "+output_name);
    }
    output << "branch,step,time,filter_applied,coordinate_norm_before,"
              "coordinate_norm_after,velocity_norm,Omega_velocity_norm,"
              "maximum_divergence,trace_rms,trace_maximum,"
              "filter_correction_velocity_norm\n";
    output << std::scientific << std::setprecision(16);

    EvolutionResult result{initial, initial};
    ExtendedPerturbationStepper stepper(extended_config, reference);
    fdm::NSCylExtendedFilterDiagnostics controlled_modal;
    for (int step = 0; step <= steps; ++step) {
        const bool apply = step == 0
            || (reorthogonalization_interval > 0
                && step%reorthogonalization_interval == 0);
        if (apply) {
            controlled_modal = filter.apply(result.controlled);
        }
        const bool log = step == 0 || step == steps
            || step%log_interval == 0 || apply;
        if (log) {
            auto diagnostic_state = result.uncontrolled;
            const auto uncontrolled_modal = filter.apply(diagnostic_state);
            write_evolution_row(
                output, "uncontrolled", step,
                (initial_time_index+step)*filter.geometry().dt, false,
                uncontrolled_modal, result.uncontrolled, filter.geometry(),
                filter.original_nr());

            if (!apply) {
                diagnostic_state = result.controlled;
                controlled_modal = filter.apply(diagnostic_state);
            }
            write_evolution_row(
                output, "controlled", step,
                (initial_time_index+step)*filter.geometry().dt, apply,
                controlled_modal, result.controlled, filter.geometry(),
                filter.original_nr());

            const Layout layout(
                filter.geometry().nr, filter.geometry().nz,
                filter.geometry().nphi);
            const double uncontrolled_norm = layout.velocity_norm(
                filter.geometry(), result.uncontrolled.data());
            const double controlled_norm = layout.velocity_norm(
                filter.geometry(), result.controlled.data());
            if (!std::isfinite(uncontrolled_norm)
                || !std::isfinite(controlled_norm)
                || uncontrolled_norm > maximum_velocity_norm
                || controlled_norm > maximum_velocity_norm) {
                throw std::runtime_error(
                    "extended evolution exceeded the velocity norm limit at "
                    "step "+std::to_string(step));
            }
        }
        if (step != steps) {
            stepper.step(result.uncontrolled);
            stepper.step(result.controlled);
        }
    }
    return result;
}

int run(const Config& config) {
    const std::string checkpoint_input = config.get(
        "checkpoint", "input", std::string());
    const std::string checkpoint_input_datatype = config.get(
        "checkpoint", "input_datatype", std::string("double"));
    const std::string spectrum_input = config.get(
        "extended", "spectrum_input", std::string());
    const std::string checkpoint_output = config.get(
        "extended", "checkpoint_output", std::string());
    const std::string diagnostics_output = config.get(
        "extended", "diagnostics_output", std::string());
    const std::string trace_output = config.get(
        "extended", "trace_output", std::string());
    const double response_condition_limit = config.get(
        "extended", "response_condition_limit", 1e12);
    const double response_regularization = config.get(
        "extended", "response_regularization", 0.0);
    const int response_basis_count = config.get(
        "extended", "response_basis_count", 1);
    const int response_cost_horizon_steps = config.get(
        "extended", "response_cost_horizon_steps",
        config.get("extended", "response_trace_horizon_steps", 0));
    const int response_cost_sample_stride = config.get(
        "extended", "response_cost_sample_stride",
        config.get("extended", "response_trace_sample_stride", 1));
    const double response_cost_ridge = config.get(
        "extended", "response_cost_ridge", 0.0);
    const std::string response_cost = config.get(
        "extended", "response_cost", std::string("boundary_trace"));
    const double control_growth_min = config.get(
        "extended", "control_growth_min", 0.0);
    const double coordinate_tolerance = config.get(
        "extended", "coordinate_tolerance", 1e-10);
    const double preservation_tolerance = config.get(
        "extended", "preservation_tolerance", 1e-12);
    const double divergence_tolerance = config.get(
        "extended", "divergence_tolerance", 1e-9);
    const double initial_perturbation_scale = config.get(
        "extended", "initial_perturbation_scale", 1.0);
    const int evolution_steps = config.get(
        "extended", "evolution_steps", 0);
    const int evolution_log_interval = config.get(
        "extended", "evolution_log_interval", 100);
    const int reorthogonalization_interval = config.get(
        "extended", "reorthogonalization_interval", 250);
    const double maximum_velocity_norm = config.get(
        "extended", "maximum_velocity_norm", 1e8);
    const std::string evolution_output = config.get(
        "extended", "evolution_output", std::string());
    const std::string evolution_checkpoint_output = config.get(
        "extended", "evolution_checkpoint_output", std::string());
    const int boundary_evolution_steps = config.get(
        "extended", "boundary_evolution_steps", 0);
    const std::string boundary_mode = config.get(
        "extended", "boundary_mode", std::string("feedback"));
    const int boundary_log_interval = config.get(
        "extended", "boundary_log_interval", 100);
    const int boundary_feedback_interval = config.get(
        "extended", "boundary_feedback_interval", 250);
    const int boundary_nonlinear_iterations = config.get(
        "extended", "boundary_nonlinear_iterations", -1);
    const int boundary_nonlinear_map_steps = config.get(
        "extended", "boundary_nonlinear_map_steps", 20000);
    const std::string boundary_evolution_output = config.get(
        "extended", "boundary_evolution_output", std::string());
    const std::string boundary_checkpoint_output = config.get(
        "extended", "boundary_checkpoint_output", std::string());
    if (checkpoint_input.empty() || spectrum_input.empty()
        || checkpoint_output.empty() || diagnostics_output.empty()
        || trace_output.empty()) {
        throw std::invalid_argument(
            "extended filter input and output paths are required");
    }
    if (!(initial_perturbation_scale > 0)
        || !std::isfinite(initial_perturbation_scale)) {
        throw std::invalid_argument(
            "extended initial_perturbation_scale must be positive");
    }

    Task original_task(config);
    const Layout original_layout(original_task);
    std::vector<T> checkpoint;
    fdm::NSCylCheckpointMetadata checkpoint_metadata;
    if (checkpoint_input_datatype == "double") {
        fdm::NSCylCheckpointStorage(checkpoint_input).load(
            checkpoint, checkpoint_metadata,
            fdm::make_ns_cyl_checkpoint_metadata<T>(config, 0));
    } else if (checkpoint_input_datatype == "float") {
        std::vector<float> source;
        fdm::NSCylCheckpointStorage(checkpoint_input).load(
            source, checkpoint_metadata,
            fdm::make_ns_cyl_checkpoint_metadata<float>(config, 0));
        checkpoint.assign(source.begin(), source.end());
    } else {
        throw std::invalid_argument(
            "checkpoint input_datatype must be 'double' or 'float'");
    }

    auto reference = couette_reference(original_task, original_layout);
    std::vector<T> original_perturbation(checkpoint.size());
    for (std::size_t index = 0; index < checkpoint.size(); ++index) {
        original_perturbation[index] = initial_perturbation_scale
            *(checkpoint[index]-reference[index]);
    }

    fdm::NSCylSpectralModeSet<T> modes;
    fdm::NSCylSpectralMetadata spectral_metadata;
    fdm::NSCylSpectralStorage(spectrum_input).load(
        modes, spectral_metadata);
    validate_domains(original_task, spectral_metadata);
    const std::size_t available_mode_count = modes.size();
    const int available_real_dimension = modes.real_dimension();
    modes = select_control_modes(modes, control_growth_min);
    if (modes.empty()) {
        throw std::runtime_error(
            "extended control growth threshold selected no modes");
    }

    Config extended_config = make_extended_config(
        spectral_metadata, response_condition_limit,
        response_regularization, response_basis_count,
        response_cost_horizon_steps, response_cost_sample_stride,
        response_cost_ridge, response_cost);
    fdm::NSCylSpectralProjector<T> projector(
        modes, spectral_metadata.condition_limit);
    ExtendedFilter filter(extended_config, std::move(projector));
    const Layout extended_layout(
        spectral_metadata.nr, spectral_metadata.nz, spectral_metadata.nphi);
    auto extended_perturbation = filter.embed_original_perturbation(
        original_perturbation);
    const auto unfiltered_perturbation = extended_perturbation;
    const double divergence_before = maximum_divergence(
        extended_perturbation, filter.geometry());
    const auto diagnostics = filter.apply(extended_perturbation, true);
    const double divergence_after = maximum_divergence(
        extended_perturbation, filter.geometry());
    std::vector<T> correction(extended_perturbation.size());
    for (std::size_t index = 0; index < correction.size(); ++index) {
        correction[index] =
            extended_perturbation[index]-unfiltered_perturbation[index];
    }
    const double correction_divergence = maximum_divergence(
        correction, filter.geometry());

    Task extended_task(extended_config);
    extended_layout.initialize_couette_state(
        extended_task, spectral_metadata.base_outer_radius);
    auto extended_reference = extended_layout.pack(extended_task);
    std::unique_ptr<fdm::NSCylSpectralFilter<T>> nonlinear_filter;
    std::unique_ptr<ExtendedNonlinearMethod> nonlinear_method;
    if (boundary_evolution_steps > 0
        && boundary_nonlinear_iterations >= 0) {
        if (boundary_nonlinear_map_steps <= 0
            || spectral_metadata.operator_steps <= 0) {
            throw std::invalid_argument(
                "extended boundary nonlinear map steps must be positive");
        }
        nonlinear_filter = std::make_unique<fdm::NSCylSpectralFilter<T>>(
            spectral_metadata.nr, spectral_metadata.nphi,
            spectral_metadata.nz,
            fdm::NSCylSpectralProjector<T>(
                modes, spectral_metadata.condition_limit));
        nonlinear_method = std::make_unique<ExtendedNonlinearMethod>(
            extended_config, extended_reference, *nonlinear_filter,
            fdm::ns_cyl_block_multipliers(modes),
            boundary_nonlinear_map_steps,
            static_cast<double>(boundary_nonlinear_map_steps)
                /spectral_metadata.operator_steps);
    }
    std::vector<T> extended_state(extended_layout.state_size);
    for (int index = 0; index < extended_layout.state_size; ++index) {
        extended_state[index] =
            extended_reference[index]+extended_perturbation[index];
    }
    extended_layout.normalize_packed_pressure(
        extended_task, extended_state.data());
    auto output_metadata = fdm::make_ns_cyl_checkpoint_metadata<T>(
        extended_config, checkpoint_metadata.time_index);
    fdm::NSCylCheckpointStorage(checkpoint_output).save(
        extended_state, output_metadata);
    write_diagnostics(diagnostics_output, diagnostics);
    write_trace(trace_output, filter.correction_boundary_velocity(),
                filter.geometry());

    if (evolution_steps > 0) {
        if (evolution_output.empty()) {
            throw std::invalid_argument(
                "extended evolution_output is required when evolution_steps "
                "is positive");
        }
        const auto evolution = run_evolution(
            extended_config, filter, extended_reference,
            unfiltered_perturbation, checkpoint_metadata.time_index,
            evolution_steps, evolution_log_interval,
            reorthogonalization_interval, maximum_velocity_norm,
            evolution_output);
        if (!evolution_checkpoint_output.empty()) {
            std::vector<T> final_state(extended_layout.state_size);
            for (int index = 0; index < extended_layout.state_size; ++index) {
                final_state[index] =
                    extended_reference[index]+evolution.controlled[index];
            }
            extended_layout.normalize_packed_pressure(
                extended_task, final_state.data());
            auto final_metadata = fdm::make_ns_cyl_checkpoint_metadata<T>(
                extended_config,
                checkpoint_metadata.time_index+evolution_steps);
            fdm::NSCylCheckpointStorage(evolution_checkpoint_output).save(
                final_state, final_metadata);
        }
        std::printf("evolution: steps=%d reorthogonalization_interval=%d "
                    "csv=%s\n",
                    evolution_steps, reorthogonalization_interval,
                    evolution_output.c_str());
    }


    if (boundary_evolution_steps > 0) {
        if (boundary_evolution_output.empty()) {
            throw std::invalid_argument(
                "extended boundary_evolution_output is required when "
                "boundary_evolution_steps is positive");
        }
        BoundaryEvolutionResult boundary;
        if (boundary_mode == "feedback") {
            boundary = run_boundary_evolution(
                config, filter, nonlinear_method.get(),
                boundary_nonlinear_iterations, reference,
                original_perturbation, checkpoint_metadata.time_index,
                boundary_evolution_steps, boundary_log_interval,
                boundary_feedback_interval, maximum_velocity_norm,
                boundary_evolution_output);
        } else if (boundary_mode == "extended_trace") {
            boundary = run_extended_trace_evolution(
                config, extended_config, filter, nonlinear_method.get(),
                boundary_nonlinear_iterations, extended_reference,
                reference, original_perturbation,
                checkpoint_metadata.time_index, boundary_evolution_steps,
                boundary_log_interval, boundary_feedback_interval,
                maximum_velocity_norm, boundary_evolution_output);
        } else {
            throw std::invalid_argument(
                "extended boundary_mode must be 'feedback' or "
                "'extended_trace'");
        }
        if (!boundary_checkpoint_output.empty()) {
            original_layout.normalize_packed_pressure(
                original_task, boundary.controlled.data());
            auto metadata = fdm::make_ns_cyl_checkpoint_metadata<T>(
                config,
                checkpoint_metadata.time_index+boundary_evolution_steps);
            fdm::NSCylCheckpointStorage(boundary_checkpoint_output).save(
                boundary.controlled, metadata);
        }
        std::printf("boundary evolution: mode=%s steps=%d "
                    "feedback_interval=%d "
                    "nonlinear_iterations=%d nonlinear_map_steps=%d "
                    "csv=%s\n",
                    boundary_mode.c_str(), boundary_evolution_steps,
                    boundary_feedback_interval,
                    boundary_nonlinear_iterations,
                    boundary_nonlinear_map_steps,
                    boundary_evolution_output.c_str());
    }

    const double coordinate_ratio = diagnostics.unstable_coordinate_norm_after
        /std::max(diagnostics.unstable_coordinate_norm_before,
                  std::numeric_limits<double>::min());
    std::printf("NSCyl extended-domain biorthogonal filter\n");
    std::printf("Omega: nr=%d r=[%.9g,%.9g]  G: nr=%d r=[%.9g,%.9g]\n",
                original_task.nr, original_task.r0, original_task.R,
                spectral_metadata.nr, spectral_metadata.r,
                spectral_metadata.R);
    std::printf("spectrum: groups=%zu blocks=%zu real_dimension=%d\n",
                modes.size(), diagnostics.blocks.size(),
                modes.real_dimension());
    std::printf("control selection: growth>=%.9e, available groups=%zu "
                "real_dimension=%d\n",
                control_growth_min, available_mode_count,
                available_real_dimension);
    std::printf("response regularization: alpha=%.9e\n",
                response_regularization);
    std::printf("response basis: count=%d cost=%s horizon=%d "
                "sample_stride=%d cost_ridge=%.9e\n",
                response_basis_count, response_cost.c_str(),
                response_cost_horizon_steps,
                response_cost_sample_stride, response_cost_ridge);
    std::printf("initial perturbation scale: %.9e\n",
                initial_perturbation_scale);
    std::printf("coordinates: before=%.9e after=%.9e ratio=%.9e\n",
                diagnostics.unstable_coordinate_norm_before,
                diagnostics.unstable_coordinate_norm_after,
                coordinate_ratio);
    std::printf("correction velocity=%.9e Omega change=%.9e\n",
                diagnostics.correction_velocity_norm,
                diagnostics.original_domain_change_norm);
    std::printf("divergence: before=%.9e after=%.9e correction=%.9e\n",
                divergence_before, divergence_after, correction_divergence);
    std::printf("checkpoint: %s\ntrace: %s\ndiagnostics: %s\n",
                checkpoint_output.c_str(), trace_output.c_str(),
                diagnostics_output.c_str());

    const bool coordinate_ok = response_regularization == 0
        ? coordinate_ratio <= coordinate_tolerance
        : diagnostics.unstable_coordinate_norm_after
            <= diagnostics.unstable_coordinate_norm_before
                *(1+coordinate_tolerance);
    const bool passed = coordinate_ok
        && diagnostics.original_domain_change_norm <= preservation_tolerance
        && correction_divergence <= divergence_tolerance
        && divergence_after <= divergence_before+divergence_tolerance;
    std::printf("RESULT: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    std::string config_name = "ns_cyl_extended_filter.ini";
    for (int i = 1; i+1 < argc; ++i) {
        if (!std::strcmp(argv[i], "-c")) {
            config_name = argv[i+1];
        }
    }
    Config config;
    config.open(config_name);
    config.rewrite(argc, argv);
    try {
        return run(config);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "extended NSCyl filter failed: %s\n",
                     error.what());
        return 1;
    }
}
