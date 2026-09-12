#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "config.h"
#include "ns_cyl_boundary_lqr.h"
#include "ns_cyl_boundary_lqr_storage.h"
#include "ns_cyl_checkpoint_storage.h"
#include "ns_cyl_fourier_energy.h"
#include "ns_cyl_spectral_storage.h"
#include "ns_cyl_state.h"
#include "ns_cyl_sycl.h"

namespace {

struct Geometry {
    int nr;
    int nz;
    int nphi;
    double r0;
    double R;
    double h1;
    double h2;
    double dr;
    double dz;
    double dphi;
    double U0;
    double Re;
    double dt;
};

Geometry make_geometry(const Config& config) {
    Geometry result;
    result.nr = config.get("ns", "nr", 32);
    result.nz = config.get("ns", "nz", 32);
    result.nphi = config.get("ns", "nphi", 32);
    result.r0 = config.get("ns", "r", M_PI/2);
    result.R = config.get("ns", "R", M_PI);
    result.h1 = config.get("ns", "h1", 0.0);
    result.h2 = config.get("ns", "h2", 10.0);
    result.U0 = config.get("ns", "u0", 1.0);
    result.Re = config.get("ns", "Re", 100.0);
    result.dt = config.get("ns", "dt", 0.001);
    result.dr = (result.R-result.r0)/result.nr;
    result.dz = (result.h2-result.h1)/result.nz;
    result.dphi = 2*M_PI/result.nphi;
    if (result.nr < 2 || result.nz <= 0 || result.nphi <= 0
        || !(result.R > result.r0) || !(result.h2 > result.h1)
        || !(result.dt > 0)) {
        throw std::invalid_argument("invalid NS cylinder SYCL geometry");
    }
    return result;
}

bool nearly_equal(double first, double second) {
    return std::abs(first-second) <= 256*std::numeric_limits<double>::epsilon()
        *std::max({1.0, std::abs(first), std::abs(second)});
}

void validate_checkpoint(
    const Geometry& geometry, const fdm::NSCylCheckpointMetadata& metadata) {
    if (metadata.scalar_type != "float64"
        || metadata.nr != geometry.nr || metadata.nz != geometry.nz
        || metadata.nphi != geometry.nphi
        || !nearly_equal(metadata.r, geometry.r0)
        || !nearly_equal(metadata.R, geometry.R)
        || !nearly_equal(metadata.h1, geometry.h1)
        || !nearly_equal(metadata.h2, geometry.h2)
        || !nearly_equal(metadata.reynolds, geometry.Re)
        || !nearly_equal(metadata.dt, geometry.dt)
        || !nearly_equal(metadata.wall_speed, geometry.U0)) {
        throw std::runtime_error(
            "checkpoint and NS cylinder SYCL geometry are incompatible");
    }
}

void validate_spectrum(
    const Geometry& geometry, const fdm::NSCylSpectralMetadata& metadata,
    bool physical) {
    const double radial_step = (metadata.R-metadata.r)/metadata.nr;
    if (metadata.scalar_type != "float64"
        || metadata.nphi != geometry.nphi || metadata.nz != geometry.nz
        || !nearly_equal(metadata.r, geometry.r0)
        || !nearly_equal(metadata.h1, geometry.h1)
        || !nearly_equal(metadata.h2, geometry.h2)
        || !nearly_equal(metadata.reynolds, geometry.Re)
        || !nearly_equal(metadata.dt, geometry.dt)
        || !nearly_equal(metadata.wall_speed, geometry.U0)
        || !nearly_equal(metadata.base_outer_radius, geometry.R)
        || !nearly_equal(radial_step, geometry.dr)
        || (physical && (metadata.nr != geometry.nr
                         || !nearly_equal(metadata.R, geometry.R)))) {
        throw std::runtime_error(
            std::string(physical ? "physical" : "extended")
            +" spectrum and NS cylinder SYCL geometry are incompatible");
    }
}

std::vector<std::pair<int, int>> selected_blocks(
    const fdm::NSCylSpectralModeSet<double>& extended,
    const fdm::NSCylSpectralModeSet<double>& physical,
    double minimum_growth) {
    std::set<std::pair<int, int>> unique;
    for (const auto* modes : {&extended, &physical}) {
        for (const auto& mode : modes->modes()) {
            if (mode.growth_rate >= minimum_growth) {
                unique.emplace(mode.m, mode.l);
            }
        }
    }
    return {unique.begin(), unique.end()};
}

std::vector<double> couette_reference(
    const Geometry& geometry,
    const fdm::NSCylStateLayout<double>& layout) {
    const auto velocity =
        fdm::make_discrete_couette_velocity<double>(geometry);
    const auto pressure =
        fdm::make_discrete_couette_pressure(geometry, velocity);
    std::vector<double> result(layout.state_size, 0.0);
    for (int i = 0; i < geometry.nphi; ++i) {
        for (int k = 0; k < geometry.nz; ++k) {
            const int plane = i*geometry.nz+k;
            for (int j = 1; j <= geometry.nr; ++j) {
                result[layout.w_offset+plane*geometry.nr+j-1] = velocity[j];
                result[layout.p_offset+plane*geometry.nr+j-1] = pressure[j];
            }
        }
    }
    return result;
}

template<typename Destination, typename Source>
std::vector<Destination> convert(const std::vector<Source>& source) {
    return {source.begin(), source.end()};
}

std::vector<double> perturbation(
    const std::vector<float>& state, const std::vector<double>& reference) {
    if (state.size() != reference.size()) {
        throw std::invalid_argument("NS cylinder SYCL state size mismatch");
    }
    std::vector<double> result(state.size());
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<double>(state[index])-reference[index];
    }
    return result;
}

double maximum_divergence(fdm::NSCylSycl<float>& state) {
    const auto u=state.ua(), v=state.va(), w=state.wa();
    double result = 0;
    for (int i = 0; i < state.nphi; ++i) {
        for (int k = 0; k < state.nz; ++k) {
            for (int j = 1; j <= state.nr; ++j) {
                const double radius = state.r0+(j-0.5)*state.dr;
                const double value =
                    ((radius+0.5*state.dr)*u(i,k,j)
                     -(radius-0.5*state.dr)*u(i,k,j-1))
                        /(radius*state.dr)
                    +(static_cast<double>(v(i,k,j))-v(i,k-1,j))/state.dz
                    +(static_cast<double>(w(i,k,j))-w(i-1,k,j))
                        /(radius*state.dphi);
                result = std::max(result, std::abs(value));
            }
        }
    }
    return result;
}

int run(const Config& config) {
    const Geometry geometry = make_geometry(config);
    const fdm::NSCylStateLayout<double> layout(
        geometry.nr, geometry.nz, geometry.nphi);
    const std::string checkpoint_input = config.get(
        "checkpoint", "input", std::string());
    const std::string extended_spectrum_input = config.get(
        "extended", "spectrum_input", std::string());
    const std::string physical_spectrum_input = config.get(
        "extended", "boundary_spectrum_input", std::string());
    const std::string evolution_output = config.get(
        "extended", "boundary_evolution_output", std::string());
    const std::string fourier_output_name = config.get(
        "extended", "boundary_fourier_output", std::string());
    const std::string checkpoint_output = config.get(
        "extended", "boundary_checkpoint_output", std::string());
    const std::string gain_input = config.get(
        "extended", "boundary_lqr_gain_input", std::string());
    const std::string gain_output = config.get(
        "extended", "boundary_lqr_gain_output", std::string());
    if (checkpoint_input.empty() || extended_spectrum_input.empty()
        || physical_spectrum_input.empty() || evolution_output.empty()) {
        throw std::invalid_argument(
            "SYCL boundary LQR input and output paths are required");
    }

    fdm::NSCylSpectralModeSet<double> extended_modes;
    fdm::NSCylSpectralModeSet<double> physical_modes;
    fdm::NSCylSpectralMetadata extended_metadata;
    fdm::NSCylSpectralMetadata physical_metadata;
    fdm::NSCylSpectralStorage(extended_spectrum_input).load(
        extended_modes, extended_metadata);
    fdm::NSCylSpectralStorage(physical_spectrum_input).load(
        physical_modes, physical_metadata);
    validate_spectrum(geometry, extended_metadata, false);
    validate_spectrum(geometry, physical_metadata, true);
    const double minimum_growth = config.get(
        "extended", "control_growth_min", 0.0);
    const auto blocks = selected_blocks(
        extended_modes, physical_modes, minimum_growth);

    const int horizon = config.get(
        "extended", "lqr_horizon_intervals", 0);
    const int feedback_interval = config.get(
        "extended", "boundary_feedback_interval", 250);
    const double control_weight = config.get(
        "extended", "lqr_control_weight", 0.0);
    const double ridge = config.get("extended", "lqr_ridge", 0.0);
    const std::string components = config.get(
        "extended", "boundary_control_components",
        std::string("tangential"));
    if (horizon <= 0 || feedback_interval <= 0) {
        throw std::invalid_argument(
            "SYCL boundary LQR horizon and feedback interval must be positive");
    }
    const auto gain_metadata =
        fdm::make_ns_cyl_boundary_lqr_gain_metadata<double>(
            config, horizon, feedback_interval,
            control_weight, ridge, components);
    std::unique_ptr<fdm::NSCylBoundaryLQR<double>> controller;
    if (!gain_input.empty()) {
        fdm::NSCylBoundaryLQRGainSet<double> gains;
        fdm::NSCylBoundaryLQRGainMetadata loaded_metadata;
        fdm::NSCylBoundaryLQRGainStorage(gain_input).load(
            gains, loaded_metadata, gain_metadata);
        std::set<std::pair<int, int>> gain_blocks;
        for (const auto& gain : gains.blocks) {
            gain_blocks.emplace(gain.m, gain.l);
        }
        if (gain_blocks != std::set<std::pair<int, int>>(
                blocks.begin(), blocks.end())) {
            throw std::runtime_error(
                "cached boundary LQR gain and selected spectra disagree");
        }
        std::printf("loading double/native boundary LQR gain: blocks=%zu "
                    "H=%d interval=%d file=%s\n", gains.blocks.size(),
                    horizon, feedback_interval, gain_input.c_str());
        controller = std::make_unique<fdm::NSCylBoundaryLQR<double>>(
            config, gains, horizon, feedback_interval,
            control_weight, ridge, components);
    } else {
        std::printf("building double/native boundary LQR: blocks=%zu H=%d "
                    "interval=%d\n", blocks.size(), horizon,
                    feedback_interval);
        controller = std::make_unique<fdm::NSCylBoundaryLQR<double>>(
            config, blocks, horizon, feedback_interval,
            control_weight, ridge, components, true);
        if (!gain_output.empty()) {
            fdm::NSCylBoundaryLQRGainStorage(gain_output).save(
                controller->cached_gain_set(), gain_metadata);
            std::printf("saved double/native boundary LQR gain: %s\n",
                        gain_output.c_str());
        }
    }

    std::vector<double> checkpoint;
    fdm::NSCylCheckpointMetadata checkpoint_metadata;
    fdm::NSCylCheckpointStorage(checkpoint_input).load(
        checkpoint, checkpoint_metadata);
    validate_checkpoint(geometry, checkpoint_metadata);
    const auto reference = couette_reference(geometry, layout);
    if (checkpoint.size() != reference.size()) {
        throw std::runtime_error("checkpoint has the wrong state size");
    }
    const double initial_scale = config.get(
        "extended", "initial_perturbation_scale", 1.0);
    if (!(initial_scale > 0) || !std::isfinite(initial_scale)) {
        throw std::invalid_argument(
            "initial perturbation scale must be positive");
    }
    std::vector<float> initial(checkpoint.size());
    for (std::size_t index = 0; index < initial.size(); ++index) {
        initial[index] = static_cast<float>(reference[index]
            +initial_scale*(checkpoint[index]-reference[index]));
    }

    sycl::queue queue{
        []() {
            for (auto& platform : sycl::platform::get_platforms()) {
                for (auto& device : platform.get_devices()) {
                    if (device.is_gpu()) {
                        return device;
                    }
                }
            }
            return sycl::device{sycl::cpu_selector_v};
        }(),
        sycl::property::queue::in_order{}};
    std::printf("SYCL device: %s\n",
                queue.get_device().get_info<sycl::info::device::name>().c_str());
    fdm::NSCylSycl<float> uncontrolled(
        queue, geometry.nr, geometry.nz, geometry.nphi,
        static_cast<float>(geometry.r0), static_cast<float>(geometry.R),
        static_cast<float>(geometry.h2-geometry.h1),
        static_cast<float>(geometry.U0), static_cast<float>(geometry.Re),
        static_cast<float>(geometry.dt));
    fdm::NSCylSycl<float> controlled(
        queue, geometry.nr, geometry.nz, geometry.nphi,
        static_cast<float>(geometry.r0), static_cast<float>(geometry.R),
        static_cast<float>(geometry.h2-geometry.h1),
        static_cast<float>(geometry.U0), static_cast<float>(geometry.Re),
        static_cast<float>(geometry.dt));
    uncontrolled.unpack_state(initial);
    controlled.unpack_state(initial);

    std::ofstream output(evolution_output);
    if (!output) {
        throw std::runtime_error(
            "cannot create SYCL boundary evolution CSV: "+evolution_output);
    }
    output << "branch,step,time,feedback_applied,velocity_norm,"
              "maximum_divergence,boundary_rms,boundary_maximum,"
              "controlled_block_velocity_norm,other_block_velocity_norm\n"
           << std::scientific << std::setprecision(16);
    std::unique_ptr<std::ofstream> fourier_output;
    std::unique_ptr<fdm::NSCylFourierVelocityEnergy<double>> fourier_energy;
    if (!fourier_output_name.empty()) {
        fourier_output = std::make_unique<std::ofstream>(fourier_output_name);
        if (!*fourier_output) {
            throw std::runtime_error(
                "cannot create SYCL Fourier-energy CSV: "+fourier_output_name);
        }
        *fourier_output << "branch,step,time,m,l,velocity_norm\n"
                        << std::scientific << std::setprecision(16);
        fourier_energy = std::make_unique<
            fdm::NSCylFourierVelocityEnergy<double>>(
                geometry.nr, geometry.nz, geometry.nphi);
    }

    const int steps = config.get(
        "extended", "boundary_evolution_steps", 0);
    const int log_interval = config.get(
        "extended", "boundary_log_interval", feedback_interval);
    const double maximum_velocity_norm = config.get(
        "extended", "maximum_velocity_norm", 100.0);
    if (steps < 0 || log_interval <= 0 || !(maximum_velocity_norm > 0)) {
        throw std::invalid_argument("invalid SYCL boundary evolution settings");
    }
    const std::size_t plane_size =
        static_cast<std::size_t>(geometry.nphi)*geometry.nz;
    std::vector<double> applied_radial(plane_size, 0.0);
    std::vector<double> applied_axial(plane_size, 0.0);
    std::vector<double> applied_azimuthal(plane_size, 0.0);
    fdm::NSCylBoundaryLQRResult<double> command;
    command.radial = applied_radial;
    command.axial = applied_axial;
    command.azimuthal = applied_azimuthal;

    for (int step = 0; step <= steps; ++step) {
        const bool feedback = step%feedback_interval == 0;
        const bool log = feedback || step == 0 || step == steps
            || step%log_interval == 0;
        std::vector<float> controlled_state;
        std::vector<double> controlled_q;
        if (feedback || log) {
            controlled_state = controlled.pack_state();
            controlled_q = perturbation(controlled_state, reference);
        }
        if (feedback) {
            command = controller->control(
                controlled_q, applied_radial, applied_axial,
                applied_azimuthal);
        }
        if (log) {
            const auto uncontrolled_state = uncontrolled.pack_state();
            const auto uncontrolled_q = perturbation(
                uncontrolled_state, reference);
            const double uncontrolled_norm = layout.velocity_norm(
                geometry, uncontrolled_q.data());
            const double controlled_norm = layout.velocity_norm(
                geometry, controlled_q.data());
            const double uncontrolled_block =
                controller->controlled_block_velocity_norm(uncontrolled_q);
            const double controlled_block =
                controller->controlled_block_velocity_norm(controlled_q);
            const double uncontrolled_other = std::sqrt(std::max(
                0.0, uncontrolled_norm*uncontrolled_norm
                    -uncontrolled_block*uncontrolled_block));
            const double controlled_other = std::sqrt(std::max(
                0.0, controlled_norm*controlled_norm
                    -controlled_block*controlled_block));
            const double time = (checkpoint_metadata.time_index+step)
                *geometry.dt;
            output << "uncontrolled," << step << ',' << time
                   << ",0," << uncontrolled_norm << ','
                   << maximum_divergence(uncontrolled)
                   << ",0,0," << uncontrolled_block << ','
                   << uncontrolled_other << '\n';
            output << "boundary," << step << ',' << time << ','
                   << (feedback ? 1 : 0) << ',' << controlled_norm << ','
                   << maximum_divergence(controlled) << ','
                   << command.rms_norm() << ',' << command.maximum_norm()
                   << ',' << controlled_block << ',' << controlled_other
                   << '\n';
            if (fourier_output) {
                for (const auto& branch : {
                         std::pair<const char*, const std::vector<double>*>{
                             "uncontrolled", &uncontrolled_q},
                         {"boundary", &controlled_q}}) {
                    const auto energies = fourier_energy->energies(
                        geometry, *branch.second);
                    for (int m = 0; m < fourier_energy->m_count(); ++m) {
                        for (int l = 0; l < fourier_energy->l_count(); ++l) {
                            *fourier_output << branch.first << ',' << step
                                << ',' << time << ',' << m << ',' << l << ','
                                << std::sqrt(std::max(0.0, energies[
                                    fourier_energy->block_index(m, l)]))
                                << '\n';
                        }
                    }
                }
            }
            if (!std::isfinite(uncontrolled_norm)
                || !std::isfinite(controlled_norm)
                || uncontrolled_norm > maximum_velocity_norm
                || controlled_norm > maximum_velocity_norm) {
                throw std::runtime_error(
                    "SYCL boundary evolution exceeded its velocity limit");
            }
            if (step == 0 || step == steps
                || step%(10*feedback_interval) == 0) {
                std::printf("step=%d t-t0=%.6g free=%.9e controlled=%.9e "
                            "wall=%.9e\n", step, step*geometry.dt,
                            uncontrolled_norm, controlled_norm,
                            command.rms_norm());
            }
        }

        if (step != steps) {
            uncontrolled.step();
            if (feedback) {
                controlled.set_outer_boundary_step_data(
                    convert<float>(applied_radial),
                    convert<float>(applied_axial),
                    convert<float>(applied_azimuthal),
                    convert<float>(command.radial),
                    convert<float>(command.axial),
                    convert<float>(command.azimuthal));
                applied_radial = command.radial;
                applied_axial = command.axial;
                applied_azimuthal = command.azimuthal;
            }
            controlled.step();
        }
    }
    queue.wait();

    if (!checkpoint_output.empty()) {
        auto final_state = controlled.pack_state();
        const fdm::NSCylStateLayout<float> float_layout(
            geometry.nr, geometry.nz, geometry.nphi);
        float_layout.normalize_packed_pressure(
            controlled, final_state.data());
        const auto metadata = fdm::make_ns_cyl_checkpoint_metadata<float>(
            config, checkpoint_metadata.time_index+steps);
        fdm::NSCylCheckpointStorage(checkpoint_output).save(
            final_state, metadata);
    }
    std::printf("SYCL boundary evolution complete: steps=%d csv=%s\n",
                steps, evolution_output.c_str());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string config_name = "ns_cyl_boundary_lqr_sycl.ini";
    for (int i = 1; i+1 < argc; ++i) {
        if (!std::strcmp(argv[i], "-c")) {
            config_name = argv[i+1];
        }
    }
    try {
        Config config;
        config.open(config_name);
        config.rewrite(argc, argv);
        return run(config);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
