#include "ns_cyl.h"

#include "ns_cyl_checkpoint_storage.h"
#include "ns_cyl_spectral_filter.h"
#include "ns_cyl_spectral_storage.h"
#include "ns_cyl_state.h"
#include "velocity_plot.h"

#include <fstream>
#include <memory>
#include <stdexcept>

using namespace fdm;
using namespace asp;
using std::vector;
using std::string;

template<typename T>
NSCylSpectralMetadata runtime_spectral_metadata(
    const Config& config, const NSCylSpectralMetadata& stored) {
    const auto current = make_ns_cyl_spectral_metadata<T>(config);
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

void require_spectral_threshold(const char* name, double requested,
                                double stored) {
    if (requested != stored) {
        throw std::invalid_argument(
            string("st:")+name+" does not match the spectral file");
    }
}

template<typename T, bool check, tensor_flag zflag>
void calc(const Config& c) {
    using namespace std::chrono;

    using Task = NSCyl<T, check, zflag>;
    Task ns(c);
    NSCylStateLayout<T> layout(ns);

    const int steps = c.get("ns", "steps", 1);
    const int plot_interval = c.get("plot", "interval", 100);
    const int png = c.get("plot", "png", 1);
    const int vtk = c.get("plot", "vtk", 0);
    const string checkpoint_input = c.get(
        "checkpoint", "input", string());
    const string checkpoint_output = c.get(
        "checkpoint", "output", string());
    bool stabilize = c.get("st", "enable", 0) != 0;
    const string spectral_input = c.get("st", "input", "ns_cyl_spectrum.nc");
    const string spectral_mode = c.get(
        "st", "mode", "unstable_eigenspace");
    const string spectral_schedule = c.get("st", "schedule", "once");
    const int legacy_ststep = c.get("st", "step", 100);
    const int spectral_start = c.get("st", "start_step", legacy_ststep);
    const int spectral_interval = c.get("st", "interval", legacy_ststep);
    const string spectral_log = c.get("st", "log", string());
    std::unique_ptr<NSCylSpectralFilter<T>> spectral_filter;
    vector<T> couette_reference;
    NSCylSpectralRemoval removal =
        NSCylSpectralRemoval::unstable_eigenspace;
    std::ofstream modal_log;
    bool one_shot_done = false;
    int i;

    if (!checkpoint_input.empty() || !checkpoint_output.empty()) {
        if constexpr (zflag != tensor_flag::periodic) {
            throw std::invalid_argument(
                "NSCyl checkpoints currently require ns:zperiod=1");
        }
    }
    if (!checkpoint_input.empty()) {
        vector<T> checkpoint_state;
        NSCylCheckpointMetadata checkpoint_metadata;
        const auto expected = make_ns_cyl_checkpoint_metadata<T>(c, 0);
        NSCylCheckpointStorage(checkpoint_input).load(
            checkpoint_state, checkpoint_metadata, expected);
        layout.unpack(ns, checkpoint_state.data());
        ns.time_index = checkpoint_metadata.time_index;
        printf("checkpoint loaded: file=%s step=%d time=%.9e\n",
               checkpoint_input.c_str(), ns.time_index,
               checkpoint_metadata.physical_time);
    }

    if (stabilize) {
        if constexpr (zflag != tensor_flag::periodic) {
            throw std::invalid_argument(
                "spectral filtering requires ns:zperiod=1");
        }
        if (spectral_start < 0) {
            throw std::invalid_argument("st:start_step must be nonnegative");
        }
        if (spectral_schedule != "once"
            && spectral_schedule != "periodic"
            && spectral_schedule != "measure_only") {
            throw std::invalid_argument(
                "st:schedule must be once, periodic, or measure_only");
        }
        if ((spectral_schedule == "periodic"
             || spectral_schedule == "measure_only")
            && spectral_interval <= 0) {
            throw std::invalid_argument("st:interval must be positive");
        }
        if (spectral_mode == "whole_fourier_blocks") {
            removal = NSCylSpectralRemoval::whole_fourier_blocks;
        } else if (spectral_mode != "unstable_eigenspace") {
            throw std::invalid_argument(
                "st:mode must be unstable_eigenspace or whole_fourier_blocks");
        }

        NSCylSpectralModeSet<T> modes;
        NSCylSpectralMetadata stored_metadata;
        const NSCylSpectralStorage storage(spectral_input);
        storage.load(modes, stored_metadata);
        const auto expected = runtime_spectral_metadata<T>(c, stored_metadata);
        storage.load(modes, stored_metadata, expected);
        require_spectral_threshold(
            "growth_tol",
            c.get("st", "growth_tol", stored_metadata.growth_tolerance),
            stored_metadata.growth_tolerance);
        require_spectral_threshold(
            "residual_tol",
            c.get("st", "residual_tol", stored_metadata.residual_tolerance),
            stored_metadata.residual_tolerance);
        require_spectral_threshold(
            "condition_limit",
            c.get("st", "condition_limit", stored_metadata.condition_limit),
            stored_metadata.condition_limit);

        if (modes.empty()) {
            printf("spectral filter: input contains no unstable modes; "
                   "state will not be changed\n");
            stabilize = false;
        } else {
            Task couette(c);
            NSCylStateLayout<T> couette_layout(couette);
            couette_layout.initialize_couette_state(couette);
            couette_reference = couette_layout.pack(couette);
            NSCylSpectralProjector<T> projector(
                modes, stored_metadata.condition_limit);
            spectral_filter = std::make_unique<NSCylSpectralFilter<T>>(
                ns.nr, ns.nphi, ns.nz, std::move(projector));
            printf("spectral filter: file=%s blocks=%zu real_dimension=%d "
                   "mode=%s schedule=%s\n",
                   spectral_input.c_str(),
                   spectral_filter->projector().blocks().size(),
                   spectral_filter->projector().real_dimension(),
                   spectral_mode.c_str(), spectral_schedule.c_str());

            if (!spectral_log.empty()) {
                modal_log.open(spectral_log);
                if (!modal_log) {
                    throw std::runtime_error(
                        "cannot open spectral diagnostics log: "+spectral_log);
                }
                modal_log << "step,time,m,l,coordinate,"
                             "coefficient_before,coefficient_after,"
                             "block_norm,removed_norm,"
                             "remaining_unstable_norm,velocity_norm,"
                             "removed_velocity_norm,filtered_velocity_norm\n";
            }
        }
    }

    velocity_plotter<T,check,typename Task::tensor_flags> plot(
        ns.dr, ns.dz, ns.dphi,
        ns.nr, ns.nz, ns.nphi,
        ns.r0, ns.R,
        ns.h1, ns.h2,
        0, 2*M_PI, true);

    plot.set_labels("R", "Z", "PHI");

    if (png || vtk) {
        plot.use(ns.u.vec, ns.v.vec, ns.w.vec);
        plot.update();
    }

    if (png) {
        //ns.plot();
        plot.plot(format("step_%07d.png", ns.time_index), ns.time_index*ns.dt);
    }
    if (vtk) {
        //ns.vtk_out();
        plot.vtk_out(format("step_%07d.vtk", ns.time_index), ns.time_index);
    }

    auto t1 = steady_clock::now();
    for (i = 0; i < steps; i++) {
        ns.step();

        if ((i+1) % plot_interval == 0 && (png || vtk)) {
            plot.update();
            if (png) {
                //ns.plot();
                plot.plot(format("step_%07d.png", ns.time_index), ns.time_index*ns.dt);
            }
            if (vtk) {
                //ns.vtk_out();
                plot.vtk_out(format("step_%07d.vtk", ns.time_index), ns.time_index);
            }
        }
        bool run_spectral_filter = false;
        if (stabilize && ns.time_index >= spectral_start) {
            if (spectral_schedule == "once") {
                run_spectral_filter = !one_shot_done;
            } else {
                run_spectral_filter =
                    (ns.time_index-spectral_start)%spectral_interval == 0;
            }
        }
        if (run_spectral_filter) {
            const auto diagnostics = spectral_schedule == "measure_only"
                ? spectral_filter->measure(ns, couette_reference, removal)
                : spectral_filter->remove(ns, couette_reference, removal);
            one_shot_done = true;
            printf("SPECTRAL_FILTER step=%d time=%.9e "
                   "perturbation=%.9e removed=%.9e remaining=%.9e "
                   "velocity=%.9e removed_velocity=%.9e "
                   "filtered_velocity=%.9e\n",
                   ns.time_index, static_cast<double>(ns.time_index*ns.dt),
                   diagnostics.packed_perturbation_norm,
                   diagnostics.removed_norm,
                   diagnostics.remaining_unstable_norm,
                   diagnostics.velocity_perturbation_norm,
                   diagnostics.removed_velocity_norm,
                   diagnostics.filtered_velocity_norm);
            for (const auto& block : diagnostics.blocks) {
                printf("SPECTRAL_BLOCK m=%d l=%d norm=%.9e removed=%.9e "
                       "remaining=%.9e\n",
                       block.m, block.l, block.block_norm,
                       block.removed_norm, block.remaining_unstable_norm);
                for (std::size_t coordinate = 0;
                     coordinate < block.coordinates_before.size();
                     ++coordinate) {
                    printf("SPECTRAL_COORD m=%d l=%d coordinate=%zu "
                           "before=%+.9e after=%+.9e\n",
                           block.m, block.l, coordinate,
                           block.coordinates_before[coordinate],
                           block.coordinates_after[coordinate]);
                    if (modal_log) {
                        modal_log << ns.time_index << ','
                                  << static_cast<double>(
                                         ns.time_index*ns.dt) << ','
                                  << block.m << ',' << block.l << ','
                                  << coordinate << ','
                                  << block.coordinates_before[coordinate]
                                  << ',' << block.coordinates_after[coordinate]
                                  << ',' << block.block_norm << ','
                                  << block.removed_norm << ','
                                  << block.remaining_unstable_norm << ','
                                  << diagnostics.velocity_perturbation_norm
                                  << ',' << diagnostics.removed_velocity_norm
                                  << ',' << diagnostics.filtered_velocity_norm
                                  << '\n';
                    }
                }
            }
        }
    }

    auto t2 = steady_clock::now();
    if (!checkpoint_output.empty()) {
        auto checkpoint_state = layout.pack(ns);
        layout.normalize_packed_pressure(ns, checkpoint_state.data());
        const auto checkpoint_metadata =
            make_ns_cyl_checkpoint_metadata<T>(c, ns.time_index);
        NSCylCheckpointStorage(checkpoint_output).save(
            checkpoint_state, checkpoint_metadata);
        printf("checkpoint saved: file=%s step=%d time=%.9e\n",
               checkpoint_output.c_str(), ns.time_index,
               checkpoint_metadata.physical_time);
    }
    auto interval = duration_cast<duration<double>>(t2 - t1);
    printf("It took me '%f' seconds\n", interval.count());
}

template<typename T, tensor_flag zflag>
void calc1(const Config& c) {
    bool check = c.get("other", "check", 0) == 1;
    if (check) {
        calc<T, true, zflag>(c);
    } else {
        calc<T, false, zflag>(c);
    }
}

template<typename T>
void calc2(const Config& c) {
    bool periodic = c.get("ns", "zperiod", 0) == 1;
    if (periodic) {
        calc1<T, tensor_flag::periodic>(c);
    } else {
        calc1<T, tensor_flag::none>(c);
    }
}

// Флетчер, том 2, страница 398
int main(int argc, char** argv) {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    string datatype = c.get("solver", "datatype", "double");

    if (datatype == "float") {
        using T = float;
        calc2<T>(c);
    } else {
        using T = double;
        calc2<T>(c);
    }

    return 0;
}
