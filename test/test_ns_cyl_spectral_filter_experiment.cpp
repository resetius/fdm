#include <algorithm>
#include <cmath>
#include <complex>
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
#include "ns_cyl_spectral_modes.h"
#include "ns_cyl_spectral_projector.h"
#include "ns_cyl_state.h"

namespace {

using T = double;
using Task = fdm::NSCyl<T, true, fdm::tensor_flag::periodic>;
using Layout = fdm::NSCylStateLayout<T>;
using Block = fdm::NSCylFourierBlockReference<T, true>;

struct CsvRow {
    std::string experiment;
    double epsilon = 0;
    std::string branch;
    int step = 0;
    double time = 0;
    double velocity_norm = 0;
    double linear_velocity_norm = std::numeric_limits<double>::quiet_NaN();
    double linear_error = std::numeric_limits<double>::quiet_NaN();
    double unstable_norm = 0;
    double coordinate = 0;
    bool filter_applied = false;
    double removed_velocity_norm = 0;
    double maximum_divergence = 0;
};

class CsvOutput {
public:
    explicit CsvOutput(const std::string& filename)
        : output_(filename) {
        if (!output_) {
            throw std::runtime_error("cannot open experiment output: "+filename);
        }
        output_ << "experiment,epsilon,branch,step,time,velocity_norm,"
                   "linear_velocity_norm,linear_error,unstable_norm,"
                   "unstable_coordinate_norm,filter_applied,"
                   "removed_velocity_norm,"
                   "max_interior_divergence\n";
        output_.setf(std::ios::scientific);
        output_.precision(16);
    }

    void write(const CsvRow& row) {
        output_ << row.experiment << ',' << row.epsilon << ',' << row.branch
                << ',' << row.step << ',' << row.time << ','
                << row.velocity_norm << ',' << row.linear_velocity_norm << ','
                << row.linear_error << ',' << row.unstable_norm << ','
                << row.coordinate << ',' << (row.filter_applied ? 1 : 0) << ','
                << row.removed_velocity_norm << ','
                << row.maximum_divergence << '\n';
    }

private:
    std::ofstream output_;
};

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

std::vector<T> couette_reference(const Config& config) {
    Task state(config);
    const Layout layout(state);
    layout.initialize_couette_state(state);
    return layout.pack(state);
}

std::vector<T> lift_block(Block& block, const std::vector<T>& coefficients,
                          const Layout& layout) {
    block.lift(coefficients.data());
    return layout.pack(block.task());
}

double coordinate_norm(
    const fdm::NSCylSpectralFilterDiagnostics& diagnostics, bool after) {
    if (diagnostics.blocks.empty()) {
        return 0;
    }
    const auto& coordinates = after
        ? diagnostics.blocks.front().coordinates_after
        : diagnostics.blocks.front().coordinates_before;
    long double result = 0;
    for (double coordinate : coordinates) {
        result += static_cast<long double>(coordinate)*coordinate;
    }
    return std::sqrt(static_cast<double>(result));
}

std::vector<T> packed_difference(Task& first, Task& second,
                                 const Layout& layout) {
    auto result = layout.pack(first);
    const auto other = layout.pack(second);
    for (int i = 0; i < layout.state_size; ++i) {
        result[i] -= other[i];
    }
    return result;
}

std::vector<T> vector_difference(const std::vector<T>& first,
                                 const std::vector<T>& second) {
    if (first.size() != second.size()) {
        throw std::invalid_argument("incompatible experiment vectors");
    }
    std::vector<T> result(first.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        result[i] = first[i]-second[i];
    }
    return result;
}

struct ScalingResult {
    double epsilon = 0;
    double final_error = 0;
    double final_relative_error = 0;
    double maximum_divergence = 0;
};

ScalingResult run_scaling_level(
    const Config& config, Block& block,
    fdm::NSCylSpectralFilter<T>& filter,
    const std::vector<T>& reference, const std::vector<T>& eigenvector,
    double epsilon, int steps, CsvOutput& csv) {
    Task perturbed(config);
    Task base(config);
    const Layout layout(perturbed);
    layout.unpack(base, reference.data());

    std::vector<T> linear_block(block.size());
    for (int i = 0; i < block.size(); ++i) {
        linear_block[i] = epsilon*eigenvector[i];
    }
    auto linear_physical = lift_block(block, linear_block, layout);
    layout.unpack_sum(perturbed, reference, linear_physical.data());

    std::vector<T> linear_image(block.size());
    ScalingResult result;
    result.epsilon = epsilon;
    for (int step = 0; step <= steps; ++step) {
        const auto nonlinear_delta = packed_difference(
            perturbed, base, layout);
        linear_physical = lift_block(block, linear_block, layout);
        const auto error = vector_difference(nonlinear_delta, linear_physical);
        const double velocity_norm =
            layout.velocity_norm(perturbed, nonlinear_delta.data());
        const double linear_norm =
            layout.velocity_norm(perturbed, linear_physical.data());
        const double linear_error =
            layout.velocity_norm(perturbed, error.data());
        const auto modal = filter.measure(perturbed, reference);

        const double divergence = maximum_interior_divergence(perturbed);
        result.maximum_divergence = std::max(
            result.maximum_divergence, divergence);
        csv.write({"epsilon_scaling", epsilon, "nonlinear", step,
                   step*perturbed.dt, velocity_norm, linear_norm, linear_error,
                   modal.removed_norm, coordinate_norm(modal, false), false,
                   0, divergence});

        if (step == steps) {
            result.final_error = linear_error;
            result.final_relative_error = linear_error/std::max(
                linear_norm, std::numeric_limits<double>::min());
            break;
        }
        perturbed.step();
        base.step();
        block.apply(linear_image.data(), linear_block.data());
        linear_block.swap(linear_image);
    }
    return result;
}

std::vector<T> make_checkpoint(const Config& config,
                               const std::vector<T>& reference,
                               const std::vector<T>& perturbation,
                               int steps) {
    Task state(config);
    const Layout layout(state);
    layout.unpack_sum(state, reference, perturbation.data());
    for (int step = 0; step < steps; ++step) {
        state.step();
    }
    return layout.pack(state);
}

struct BranchResult {
    double immediate_ratio = std::numeric_limits<double>::quiet_NaN();
    double maximum_divergence = 0;
};

BranchResult run_branch(const Config& config,
                        fdm::NSCylSpectralFilter<T>& filter,
                        const std::vector<T>& reference,
                        const std::vector<T>& checkpoint,
                        const std::string& name, double epsilon,
                        int checkpoint_step, int steps,
                        int periodic_interval, CsvOutput& csv) {
    Task state(config);
    const Layout layout(state);
    layout.unpack(state, checkpoint.data());
    state.time_index = checkpoint_step;
    BranchResult result;

    for (int step = 0; step <= steps; ++step) {
        const bool apply_once = name == "once" && step == 0;
        const bool apply_periodic = name == "periodic"
            && step%periodic_interval == 0;
        const bool apply = apply_once || apply_periodic;
        const auto diagnostics = apply
            ? filter.remove(state, reference)
            : filter.measure(state, reference);
        const double velocity_norm = apply
            ? diagnostics.filtered_velocity_norm
            : diagnostics.velocity_perturbation_norm;
        const double unstable_norm = apply
            ? diagnostics.remaining_unstable_norm
            : diagnostics.removed_norm;
        if (apply && step == 0) {
            result.immediate_ratio =
                diagnostics.remaining_unstable_norm/std::max(
                diagnostics.removed_norm,
                std::numeric_limits<double>::min());
        }

        const double divergence = maximum_interior_divergence(state);
        result.maximum_divergence = std::max(
            result.maximum_divergence, divergence);

        csv.write({"filter_branches", epsilon, name, step,
                   (checkpoint_step+step)*state.dt, velocity_norm,
                   std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::quiet_NaN(), unstable_norm,
                   coordinate_norm(diagnostics, apply), apply,
                   apply ? diagnostics.removed_velocity_norm : 0,
                   divergence});

        if (step != steps) {
            state.step();
        }
    }
    return result;
}

int run(const Config& config) {
    const int m = config.get("experiment", "m", 0);
    const int l = config.get("experiment", "l", 3);
    const int scaling_steps = config.get("experiment", "scaling_steps", 32);
    const int epsilon_levels = config.get("experiment", "epsilon_levels", 3);
    const double epsilon = config.get("experiment", "epsilon", 1e-3);
    const int checkpoint_step = config.get(
        "experiment", "checkpoint_step", 64);
    const int branch_steps = config.get("experiment", "branch_steps", 64);
    const int periodic_interval = config.get(
        "experiment", "periodic_interval", 8);
    const double minimum_order = config.get(
        "experiment", "minimum_quadratic_order", 1.8);
    const double filter_tolerance = config.get(
        "experiment", "filter_tolerance", 1e-10);
    const double divergence_tolerance = config.get(
        "experiment", "divergence_tolerance", 1e-10);
    const std::string output = config.get(
        "experiment", "output", "ns_cyl_spectral_filter_experiment.csv");
    const std::string checkpoint_output = config.get(
        "experiment", "checkpoint_output", std::string());
    if (scaling_steps <= 0 || epsilon_levels < 2 || !(epsilon > 0)
        || checkpoint_step < 0 || branch_steps <= 0
        || periodic_interval <= 0) {
        throw std::invalid_argument("invalid spectral filter experiment setup");
    }

    Block block(config, m, l);
    const auto spectrum = fdm::solve_ns_cyl_dense_block(
        block, config.get("ns", "dt", 0.001),
        config.get("spectral", "growth_tol", 1e-8),
        config.get("spectral", "residual_tol", 1e-10));
    const auto leading = std::max_element(
        spectrum.modes.begin(), spectrum.modes.end(),
        [](const auto& a, const auto& b) {
            return a.growth_rate < b.growth_rate;
        });
    if (leading == spectrum.modes.end() || !leading->filterable_unstable()) {
        throw std::runtime_error("selected block has no accepted unstable mode");
    }

    fdm::NSCylSpectralModeSet<T> modes;
    modes.append_filterable_mode(*leading);
    fdm::NSCylSpectralFilter<T> filter(
        block.task().nr, block.task().nphi, block.task().nz,
        fdm::NSCylSpectralProjector<T>(
            modes, config.get("spectral", "condition_limit", 1e10)));
    const auto reference = couette_reference(config);
    const Layout layout(block.task());
    std::vector<T> eigenvector(
        leading->right_columns.begin(),
        leading->right_columns.begin()+block.size());
    CsvOutput csv(output);

    printf("spectral filter nonlinear validation\n");
    printf("block: m=%d l=%d size=%d phases=%d\n",
           m, l, block.size(), block.phase_count());
    printf("leading: mu=(%.16e,%+.16e) growth=%+.9e columns=%d\n",
           leading->multiplier.real(), leading->multiplier.imag(),
           leading->growth_rate, leading->column_count);

    std::vector<ScalingResult> scaling;
    bool passed = true;
    for (int level = 0; level < epsilon_levels; ++level) {
        const double level_epsilon = std::ldexp(epsilon, -level);
        scaling.push_back(run_scaling_level(
            config, block, filter, reference, eigenvector,
            level_epsilon, scaling_steps, csv));
        printf("SCALING epsilon=%.9e error=%.9e relative=%.9e\n",
               scaling.back().epsilon, scaling.back().final_error,
               scaling.back().final_relative_error);
        passed = passed
            && scaling.back().maximum_divergence <= divergence_tolerance;
    }

    for (std::size_t i = 1; i < scaling.size(); ++i) {
        const double order = std::log(
            scaling[i-1].final_error/scaling[i].final_error)/std::log(2.0);
        printf("SCALING_ORDER epsilon_hi=%.9e epsilon_lo=%.9e order=%.6f\n",
               scaling[i-1].epsilon, scaling[i].epsilon, order);
        passed = passed && std::isfinite(order) && order >= minimum_order;
    }

    std::vector<T> initial_block(block.size());
    for (int i = 0; i < block.size(); ++i) {
        initial_block[i] = epsilon*eigenvector[i];
    }
    const auto initial_perturbation = lift_block(block, initial_block, layout);
    auto checkpoint = make_checkpoint(
        config, reference, initial_perturbation, checkpoint_step);
    if (!checkpoint_output.empty()) {
        layout.normalize_packed_pressure(block.task(), checkpoint.data());
        const auto metadata = fdm::make_ns_cyl_checkpoint_metadata<T>(
            config, checkpoint_step);
        const fdm::NSCylCheckpointStorage storage(checkpoint_output);
        storage.save(checkpoint, metadata);
        std::vector<T> reloaded;
        fdm::NSCylCheckpointMetadata reloaded_metadata;
        storage.load(reloaded, reloaded_metadata,
                     fdm::make_ns_cyl_checkpoint_metadata<T>(config, 0));
        if (reloaded_metadata.time_index != checkpoint_step
            || reloaded != checkpoint) {
            throw std::runtime_error("checkpoint round-trip changed branch state");
        }
        checkpoint = std::move(reloaded);
        printf("checkpoint: %s step=%d\n",
               checkpoint_output.c_str(), checkpoint_step);
    }
    const auto unfiltered = run_branch(
        config, filter, reference, checkpoint, "unfiltered", epsilon,
        checkpoint_step, branch_steps, periodic_interval, csv);
    const auto once = run_branch(
        config, filter, reference, checkpoint, "once", epsilon,
        checkpoint_step, branch_steps, periodic_interval, csv);
    const auto periodic = run_branch(
        config, filter, reference, checkpoint, "periodic", epsilon,
        checkpoint_step, branch_steps, periodic_interval, csv);
    printf("FILTER_RATIO once=%.9e periodic=%.9e tolerance=%.9e\n",
           once.immediate_ratio, periodic.immediate_ratio, filter_tolerance);
    const double branch_divergence = std::max({
        unfiltered.maximum_divergence,
        once.maximum_divergence,
        periodic.maximum_divergence});
    printf("MAX_DIVERGENCE value=%.9e tolerance=%.9e\n",
           branch_divergence, divergence_tolerance);
    passed = passed && std::isfinite(once.immediate_ratio)
        && std::isfinite(periodic.immediate_ratio)
        && once.immediate_ratio <= filter_tolerance
        && periodic.immediate_ratio <= filter_tolerance
        && branch_divergence <= divergence_tolerance;

    printf("output: %s\n", output.c_str());
    printf("RESULT: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 2;
}

} // namespace

int main(int argc, char** argv) {
    std::string config_name = "ns_cyl_spectral_filter_experiment.ini";
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
        std::fprintf(stderr, "spectral filter experiment failed: %s\n",
                     error.what());
        return 1;
    }
}
