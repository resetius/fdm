#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "ns_cyl.h"
#include "ns_cyl_checkpoint_storage.h"
#include "ns_cyl_fourier_block.h"
#include "ns_cyl_spectral_filter.h"
#include "ns_cyl_spectral_storage.h"
#include "ns_cyl_state.h"

namespace {

using T = double;
using Task = fdm::NSCyl<T, true, fdm::tensor_flag::periodic>;
using Layout = fdm::NSCylStateLayout<T>;
using Projector = fdm::NSCylSpectralProjector<T>;
using Filter = fdm::NSCylSpectralFilter<T>;

fdm::NSCylSpectralMetadata runtime_spectral_metadata(
    const Config& config, const fdm::NSCylSpectralMetadata& stored) {
    const auto current = fdm::make_ns_cyl_spectral_metadata<T>(config);
    auto expected = stored;
    expected.scalar_type = current.scalar_type;
    expected.nr = current.nr;
    expected.nphi = current.nphi;
    expected.nz = current.nz;
    expected.radial_size = current.radial_size;
    expected.u_offset = current.u_offset;
    expected.v_offset = current.v_offset;
    expected.w_offset = current.w_offset;
    expected.p_offset = current.p_offset;
    expected.r = current.r;
    expected.R = current.R;
    expected.h1 = current.h1;
    expected.h2 = current.h2;
    expected.reynolds = current.reynolds;
    expected.dt = current.dt;
    expected.wall_speed = current.wall_speed;
    return expected;
}

double cell_divergence(Task& state, int i, int k, int j) {
    const double radius = state.r0+(j-0.5)*state.dr;
    return ((radius+0.5*state.dr)*state.u[i][k][j]
            -(radius-0.5*state.dr)*state.u[i][k][j-1])
            /(radius*state.dr)
        +(state.v[i][k][j]-state.v[i][k-1][j])/state.dz
        +(state.w[i][k][j]-state.w[i-1][k][j])
            /(radius*state.dphi);
}

double maximum_interior_divergence(Task& state) {
    double result = 0;
    for (int i = 0; i < state.nphi; ++i) {
        for (int k = 0; k < state.nz; ++k) {
            for (int j = 2; j < state.nr; ++j) {
                result = std::max(
                    result, std::abs(cell_divergence(state, i, k, j)));
            }
        }
    }
    return result;
}

double maximum_radial_boundary_residual(Task& state) {
    double result = 0;
    for (int i = 0; i < state.nphi; ++i) {
        for (int k = 0; k < state.nz; ++k) {
            result = std::max(result, std::abs(state.u[i][k][0]));
            result = std::max(result, std::abs(state.u[i][k][state.nr]));
            result = std::max(result, std::abs(
                0.5*(state.v[i][k][0]+state.v[i][k][1])));
            result = std::max(result, std::abs(
                0.5*(state.v[i][k][state.nr]
                     +state.v[i][k][state.nr+1])));
            result = std::max(result, std::abs(
                0.5*(state.w[i][k][0]+state.w[i][k][1])-state.U0));
            result = std::max(result, std::abs(
                0.5*(state.w[i][k][state.nr]
                     +state.w[i][k][state.nr+1])));
        }
    }
    return result;
}

double taylor_vortex_norm(Task& state) {
    const Layout layout(state);
    const auto packed = layout.pack(state);
    return layout.taylor_vortex_norm(state, packed.data());
}

struct TorqueFlux {
    double inner = 0;
    double outer = 0;
};

TorqueFlux viscous_torque_flux(Task& state) {
    long double inner_shear = 0;
    long double outer_shear = 0;
    const long double samples = state.nphi*state.nz;
    for (int i = 0; i < state.nphi; ++i) {
        for (int k = 0; k < state.nz; ++k) {
            inner_shear += (state.w[i][k][1]-state.U0)/(0.5*state.dr)
                -state.U0/state.r0;
            outer_shear += -state.w[i][k][state.nr]/(0.5*state.dr);
        }
    }
    const long double height = state.h2-state.h1;
    TorqueFlux result;
    result.inner = static_cast<double>(
        2*M_PI*height*state.r0*state.r0*inner_shear/(samples*state.Re));
    result.outer = static_cast<double>(
        2*M_PI*height*state.R*state.R*outer_shear/(samples*state.Re));
    return result;
}

std::vector<T> couette_reference(Task& state, const Layout& layout) {
    layout.initialize_couette_state(state);
    return layout.pack(state);
}

std::vector<T> make_multimode_seed(
    const Config& config, const Projector& projector,
    const Layout& layout, double requested_norm) {
    std::vector<T> perturbation(layout.state_size, 0);
    int global_coordinate = 0;
    for (const auto& block_projector : projector.blocks()) {
        fdm::NSCylFourierBlockReference<T, true> block(
            config, block_projector.m(), block_projector.l());
        std::vector<T> block_state(block_projector.block_size(), 0);
        for (const auto& basis : block_projector.right_basis()) {
            const T coefficient = static_cast<T>(
                0.75+0.25*std::cos((global_coordinate+1)*1.61803398875));
            for (int row = 0; row < block_projector.block_size(); ++row) {
                block_state[row] += coefficient*basis[row];
            }
            ++global_coordinate;
        }
        block.lift(block_state.data());
        const auto physical_block = layout.pack(block.task());
        for (int index = 0; index < layout.state_size; ++index) {
            perturbation[index] += physical_block[index];
        }
    }

    Task geometry(config);
    layout.normalize_packed_pressure(geometry, perturbation.data());
    const double norm = layout.velocity_norm(geometry, perturbation.data());
    if (!(norm > 0) || !std::isfinite(norm)) {
        throw std::runtime_error("multimode seed has zero or non-finite norm");
    }
    const T scale = static_cast<T>(requested_norm/norm);
    for (T& value : perturbation) {
        value *= scale;
    }
    return perturbation;
}

class CsvOutput {
public:
    explicit CsvOutput(const std::string& filename)
        : output_(filename) {
        if (!output_) {
            throw std::runtime_error("cannot open developed-flow CSV: "+filename);
        }
        output_ << "branch,local_step,time,m,l,coordinate,"
                   "coefficient_before,coefficient_after,filter_applied,"
                   "unstable_norm_before,unstable_norm_after,velocity_norm,"
                   "taylor_vortex_norm,alpha,divergence,boundary_residual,"
                   "inner_torque_flux,outer_torque_flux,removed_velocity_norm\n";
        output_.setf(std::ios::scientific);
        output_.precision(16);
    }

    void write(const std::string& branch, int local_step, double time,
               const fdm::NSCylSpectralFilterDiagnostics& modal,
               bool filter_applied, double velocity_norm,
               double vortex_norm, double alpha, double divergence,
               double boundary_residual, const TorqueFlux& torque) {
        for (const auto& block : modal.blocks) {
            for (std::size_t coordinate = 0;
                 coordinate < block.coordinates_before.size(); ++coordinate) {
                output_ << branch << ',' << local_step << ',' << time << ','
                        << block.m << ',' << block.l << ',' << coordinate << ','
                        << block.coordinates_before[coordinate] << ','
                        << block.coordinates_after[coordinate] << ','
                        << (filter_applied ? 1 : 0) << ','
                        << modal.removed_norm << ','
                        << modal.remaining_unstable_norm << ','
                        << velocity_norm << ',' << vortex_norm << ',' << alpha
                        << ',' << divergence << ',' << boundary_residual << ','
                        << torque.inner << ',' << torque.outer << ','
                        << modal.removed_velocity_norm << '\n';
            }
        }
    }

private:
    std::ofstream output_;
};

struct DecayRate {
    bool initialized = false;
    double time = 0;
    double norm = 0;

    double update(double current_time, double current_norm) {
        double result = std::numeric_limits<double>::quiet_NaN();
        if (initialized && current_time > time && norm > 0 && current_norm > 0) {
            result = (std::log(norm)-std::log(current_norm))
                /(current_time-time);
        }
        initialized = true;
        time = current_time;
        norm = current_norm;
        return result;
    }
};

struct SampleResult {
    double velocity_norm = 0;
    double unstable_norm = 0;
    double unstable_norm_before = 0;
    double unstable_norm_after = 0;
    double taylor_norm = 0;
    double divergence = 0;
    double boundary_residual = 0;
    double removed_velocity_norm = 0;
    double decay_rate = std::numeric_limits<double>::quiet_NaN();
};

SampleResult sample(Task& state, Filter& filter,
                    const std::vector<T>& reference, bool apply_filter,
                    const std::string& branch, int local_step,
                    DecayRate& decay, CsvOutput& csv) {
    // step() applies wall ghosts at the beginning of a step. Refresh them
    // before reporting boundary residuals for the newly updated interior.
    state.apply_boundary_conditions();
    const auto modal = apply_filter
        ? filter.remove(state, reference)
        : filter.measure(state, reference);
    SampleResult result;
    result.velocity_norm = apply_filter
        ? modal.filtered_velocity_norm
        : modal.velocity_perturbation_norm;
    result.unstable_norm = apply_filter
        ? modal.remaining_unstable_norm
        : modal.removed_norm;
    result.unstable_norm_before = modal.removed_norm;
    result.unstable_norm_after = modal.remaining_unstable_norm;
    result.taylor_norm = taylor_vortex_norm(state);
    result.divergence = maximum_interior_divergence(state);
    result.boundary_residual = maximum_radial_boundary_residual(state);
    result.removed_velocity_norm = modal.removed_velocity_norm;
    const double time = state.time_index*state.dt;
    const double alpha = decay.update(time, result.velocity_norm);
    result.decay_rate = alpha;
    csv.write(branch, local_step, time, modal, apply_filter,
              result.velocity_norm, result.taylor_norm, alpha,
              result.divergence, result.boundary_residual,
              viscous_torque_flux(state));
    return result;
}

struct BranchResult {
    double immediate_ratio = std::numeric_limits<double>::quiet_NaN();
    SampleResult final;
    double maximum_divergence = 0;
    double maximum_boundary_residual = 0;
};

BranchResult run_branch(
    const Config& config, Filter& filter, const std::vector<T>& reference,
    const std::vector<T>& checkpoint, int checkpoint_step,
    const std::string& name, int steps, int log_interval,
    int periodic_interval, CsvOutput& csv) {
    Task state(config);
    const Layout layout(state);
    layout.unpack(state, checkpoint.data());
    state.time_index = checkpoint_step;
    DecayRate decay;
    BranchResult result;

    for (int step = 0; step <= steps; ++step) {
        const bool apply_filter = (name == "once" && step == 0)
            || (name == "periodic" && step%periodic_interval == 0);
        const bool log = step == 0 || step == steps
            || step%log_interval == 0 || apply_filter;
        if (log) {
            const SampleResult current = sample(
                state, filter, reference, apply_filter,
                name, step, decay, csv);
            if (apply_filter && step == 0) {
                result.immediate_ratio = current.unstable_norm_after/std::max(
                    current.unstable_norm_before,
                    std::numeric_limits<double>::min());
            }
            result.maximum_divergence = std::max(
                result.maximum_divergence, current.divergence);
            result.maximum_boundary_residual = std::max(
                result.maximum_boundary_residual, current.boundary_residual);
            if (step == steps) {
                result.final = current;
            }
        }
        if (step != steps) {
            state.step();
        }
    }
    return result;
}

int run(const Config& config) {
    const std::string spectrum_input = config.get(
        "developed", "spectrum_input", std::string());
    const std::string checkpoint_output = config.get(
        "developed", "checkpoint_output", std::string());
    const std::string checkpoint_input = config.get(
        "developed", "checkpoint_input", std::string());
    const std::string csv_output = config.get(
        "developed", "output", "ns_cyl_spectral_filter_developed.csv");
    const double seed_norm = config.get(
        "developed", "seed_velocity_norm", 1e-2);
    const int develop_steps = config.get(
        "developed", "develop_steps", 20000);
    const int develop_log_interval = config.get(
        "developed", "develop_log_interval", 500);
    const int branch_steps = config.get(
        "developed", "branch_steps", 5000);
    const int branch_log_interval = config.get(
        "developed", "branch_log_interval", 100);
    const int periodic_interval = config.get(
        "developed", "periodic_interval", 250);
    const double minimum_remainder_fraction = config.get(
        "developed", "minimum_remainder_fraction", 1e-3);
    const double filter_tolerance = config.get(
        "developed", "filter_tolerance", 1e-9);
    const double divergence_tolerance = config.get(
        "developed", "divergence_tolerance", 1e-9);
    const double boundary_tolerance = config.get(
        "developed", "boundary_tolerance", 1e-12);
    const double maximum_final_decay_rate = config.get(
        "developed", "maximum_final_absolute_decay_rate", 1e-2);
    const double minimum_taylor_norm = config.get(
        "developed", "minimum_taylor_vortex_norm", 1e-1);
    const double minimum_once_reexcitation = config.get(
        "developed", "minimum_once_reexcitation", 1e-4);
    const double maximum_periodic_to_once_ratio = config.get(
        "developed", "maximum_periodic_to_once_ratio", 1e-8);
    if (spectrum_input.empty() || checkpoint_output.empty()
        || !(seed_norm > 0) || develop_steps <= 0
        || develop_log_interval <= 0 || branch_steps <= 0
        || branch_log_interval <= 0 || periodic_interval <= 0
        || branch_steps%periodic_interval != 0
        || minimum_remainder_fraction < 0 || filter_tolerance < 0
        || divergence_tolerance < 0 || boundary_tolerance < 0
        || maximum_final_decay_rate < 0 || minimum_taylor_norm < 0
        || minimum_once_reexcitation < 0
        || maximum_periodic_to_once_ratio < 0) {
        throw std::invalid_argument("invalid developed-flow experiment setup");
    }

    fdm::NSCylSpectralModeSet<T> modes;
    fdm::NSCylSpectralMetadata stored_metadata;
    const fdm::NSCylSpectralStorage spectral_storage(spectrum_input);
    spectral_storage.load(modes, stored_metadata);
    spectral_storage.load(
        modes, stored_metadata,
        runtime_spectral_metadata(config, stored_metadata));
    if (modes.empty()) {
        throw std::runtime_error("spectrum contains no unstable modes");
    }

    Projector projector(modes, stored_metadata.condition_limit);
    Task state(config);
    const Layout layout(state);
    const auto reference = couette_reference(state, layout);
    if (checkpoint_input.empty()) {
        const auto seed = make_multimode_seed(
            config, projector, layout, seed_norm);
        layout.unpack_sum(state, reference, seed.data());
    } else {
        std::vector<T> initial_state;
        fdm::NSCylCheckpointMetadata initial_metadata;
        fdm::NSCylCheckpointStorage(checkpoint_input).load(
            initial_state, initial_metadata,
            fdm::make_ns_cyl_checkpoint_metadata<T>(config, 0));
        layout.unpack(state, initial_state.data());
        state.time_index = initial_metadata.time_index;
    }
    const int initial_time_index = state.time_index;
    Filter filter(state.nr, state.nphi, state.nz, projector);
    CsvOutput csv(csv_output);
    DecayRate development_decay;

    printf("developed spectral-filter experiment\n");
    printf("spectrum: groups=%zu blocks=%zu real_dimension=%d\n",
           modes.size(), projector.blocks().size(), projector.real_dimension());
    printf("grid: nr=%d nphi=%d nz=%d Re=%.9g dt=%.9g\n",
           state.nr, state.nphi, state.nz, state.Re, state.dt);
    if (checkpoint_input.empty()) {
        printf("seed velocity norm: %.9e\n", seed_norm);
    } else {
        printf("continued from: %s step=%d\n",
               checkpoint_input.c_str(), initial_time_index);
    }

    SampleResult developed;
    double development_maximum_divergence = 0;
    double development_maximum_boundary_residual = 0;
    for (int step = 0; step <= develop_steps; ++step) {
        if (step == 0 || step == develop_steps
            || step%develop_log_interval == 0) {
            developed = sample(
                state, filter, reference, false, "develop", step,
                development_decay, csv);
            printf("DEVELOP local_step=%d step=%d velocity=%.9e unstable=%.9e "
                   "taylor=%.9e alpha=%+.3e div=%.3e\n",
                   step, state.time_index, developed.velocity_norm,
                   developed.unstable_norm,
                   developed.taylor_norm, developed.decay_rate,
                   developed.divergence);
            development_maximum_divergence = std::max(
                development_maximum_divergence, developed.divergence);
            development_maximum_boundary_residual = std::max(
                development_maximum_boundary_residual,
                developed.boundary_residual);
        }
        if (step != develop_steps) {
            state.step();
        }
    }

    auto checkpoint = layout.pack(state);
    layout.normalize_packed_pressure(state, checkpoint.data());
    const auto checkpoint_metadata =
        fdm::make_ns_cyl_checkpoint_metadata<T>(config, state.time_index);
    const fdm::NSCylCheckpointStorage checkpoint_storage(checkpoint_output);
    checkpoint_storage.save(checkpoint, checkpoint_metadata);
    std::vector<T> reloaded;
    fdm::NSCylCheckpointMetadata reloaded_metadata;
    checkpoint_storage.load(
        reloaded, reloaded_metadata,
        fdm::make_ns_cyl_checkpoint_metadata<T>(config, 0));
    if (reloaded != checkpoint
        || reloaded_metadata.time_index != state.time_index) {
        throw std::runtime_error("developed checkpoint round-trip changed state");
    }

    Task checkpoint_state(config);
    layout.unpack(checkpoint_state, checkpoint.data());
    checkpoint_state.time_index = reloaded_metadata.time_index;
    const auto checkpoint_modal = filter.measure(checkpoint_state, reference);
    const double remainder_fraction = checkpoint_modal.filtered_velocity_norm
        /std::max(checkpoint_modal.velocity_perturbation_norm,
                  std::numeric_limits<double>::min());
    printf("CHECKPOINT step=%d velocity=%.9e unstable_velocity=%.9e "
           "remainder=%.9e fraction=%.9e file=%s\n",
           reloaded_metadata.time_index,
           checkpoint_modal.velocity_perturbation_norm,
           checkpoint_modal.removed_velocity_norm,
           checkpoint_modal.filtered_velocity_norm, remainder_fraction,
           checkpoint_output.c_str());

    const auto unfiltered = run_branch(
        config, filter, reference, reloaded, reloaded_metadata.time_index,
        "unfiltered", branch_steps, branch_log_interval,
        periodic_interval, csv);
    const auto once = run_branch(
        config, filter, reference, reloaded, reloaded_metadata.time_index,
        "once", branch_steps, branch_log_interval,
        periodic_interval, csv);
    const auto periodic = run_branch(
        config, filter, reference, reloaded, reloaded_metadata.time_index,
        "periodic", branch_steps, branch_log_interval,
        periodic_interval, csv);

    printf("BRANCH unfiltered velocity=%.9e unstable=%.9e taylor=%.9e\n",
           unfiltered.final.velocity_norm, unfiltered.final.unstable_norm,
           unfiltered.final.taylor_norm);
    printf("BRANCH once velocity=%.9e unstable=%.9e taylor=%.9e "
           "immediate_ratio=%.9e\n",
           once.final.velocity_norm, once.final.unstable_norm,
           once.final.taylor_norm, once.immediate_ratio);
    printf("BRANCH periodic velocity=%.9e unstable=%.9e taylor=%.9e "
           "immediate_ratio=%.9e\n",
           periodic.final.velocity_norm, periodic.final.unstable_norm,
           periodic.final.taylor_norm, periodic.immediate_ratio);

    const double maximum_divergence = std::max({
        development_maximum_divergence, unfiltered.maximum_divergence,
        once.maximum_divergence, periodic.maximum_divergence});
    const double maximum_boundary_residual = std::max({
        development_maximum_boundary_residual,
        unfiltered.maximum_boundary_residual,
        once.maximum_boundary_residual,
        periodic.maximum_boundary_residual});
    const double periodic_to_once = periodic.final.unstable_norm/std::max(
        once.final.unstable_norm, std::numeric_limits<double>::min());
    const bool passed = std::isfinite(remainder_fraction)
        && remainder_fraction >= minimum_remainder_fraction
        && std::isfinite(developed.decay_rate)
        && std::abs(developed.decay_rate) <= maximum_final_decay_rate
        && developed.taylor_norm >= minimum_taylor_norm
        && once.immediate_ratio <= filter_tolerance
        && periodic.immediate_ratio <= filter_tolerance
        && once.final.unstable_norm >= minimum_once_reexcitation
        && periodic_to_once <= maximum_periodic_to_once_ratio
        && maximum_divergence <= divergence_tolerance
        && maximum_boundary_residual <= boundary_tolerance;
    printf("VALIDATION remainder_fraction=%.9e minimum=%.9e "
           "final_alpha=%+.9e max_abs_alpha=%.9e "
           "periodic_to_once=%.9e max_ratio=%.9e "
           "max_divergence=%.9e max_boundary=%.9e\n",
           remainder_fraction, minimum_remainder_fraction,
           developed.decay_rate, maximum_final_decay_rate,
           periodic_to_once, maximum_periodic_to_once_ratio,
           maximum_divergence, maximum_boundary_residual);
    printf("output: %s\n", csv_output.c_str());
    printf("RESULT: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    std::string config_name = "ns_cyl_spectral_filter_developed.ini";
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
        std::fprintf(stderr, "developed spectral-filter experiment failed: %s\n",
                     error.what());
        return 1;
    }
}
