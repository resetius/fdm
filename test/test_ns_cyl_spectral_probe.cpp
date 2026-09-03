#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "arpack_solver.h"
#include "config.h"
#include "ns_cyl_fourier_block.h"
#include "ns_cyl_spectral_modes.h"
#include "ns_cyl_spectral_projector.h"
#include "ns_cyl_spectral_storage.h"

using fdm::arpack_solver;
using fdm::NSCylFourierBlockReference;
using std::complex;
using std::string;
using std::vector;

namespace {

struct BlockIndex {
    int m;
    int l;
};

template<typename T>
struct ProbeResult {
    BlockIndex block{};
    int radial_size = 0;
    int phase_count = 0;
    int arpack_size = 0;
    int nev = 0;
    int ncv = 0;
    int operator_calls = 0;
    int arpack_info = 0;
    int arpack_iterations = 0;
    int arpack_nconv = 0;
    int arpack_starts = 0;
    int arpack_lr_starts = 0;
    int arpack_failed_starts = 0;
    double max_leakage = 0;
    bool candidate = false;
    bool guard_reached = false;
    bool dense_computed = false;
    string error;
    vector<complex<T>> eigenvalues;
    fdm::NSCylDenseBlockSpectrum<T> dense_spectrum;
};

template<typename T>
vector<int> sorted_indices(const vector<complex<T>>& eigenvalues) {
    vector<int> indices(eigenvalues.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return std::abs(eigenvalues[a]) > std::abs(eigenvalues[b]);
    });
    return indices;
}

template<typename T>
double residual_tolerance(const Config& config) {
    const double roundoff_floor =
        64.0*static_cast<double>(std::numeric_limits<T>::epsilon());
    return std::max(roundoff_floor,
        config.get("spectral", "residual_tol", 1e-10));
}

template<typename T>
ProbeResult<T> probe_block(const Config& config, BlockIndex index) {
    using Block = NSCylFourierBlockReference<T, false>;
    ProbeResult<T> result;
    result.block = index;

    const int operator_steps = config.get("spectral", "operator_steps", 1);
    std::unique_ptr<Block> block;

    // FFTW planning is not generally thread-safe. Each block owns its plans,
    // but their construction is serialized when the outer probe uses OpenMP.
#ifdef _OPENMP
#pragma omp critical(fdm_ns_cyl_probe_fft_planning)
#endif
    {
        block = std::make_unique<Block>(config, index.m, index.l,
                                        operator_steps);
    }

    result.radial_size = block->radial_size();
    result.phase_count = block->phase_count();
    result.arpack_size = block->size();

    const int n = block->size();
    const int largest_valid_nev = n-2;
    int nev = std::min(config.get("spectral", "nev", 4), largest_valid_nev);
    const int max_nev = std::min(
        config.get("spectral", "max_nev", 32), largest_valid_nev);
    const int requested_ncv = config.get("spectral", "ncv", 0);
    const int stable_guard = config.get("spectral", "stable_guard", 4);
    const int maxit = config.get("spectral", "maxit", 10000);
    const int residual_seed = config.get("spectral", "residual_seed", 0);
    const int probe_starts = std::max(
        1, config.get("spectral", "probe_starts", 3));
    const int probe_lr_starts = std::max(
        0, config.get("spectral", "probe_lr_starts", 1));
    const double default_tolerance = std::max(
        1e-8, 8.0*static_cast<double>(std::numeric_limits<T>::epsilon()));
    const T tolerance = static_cast<T>(
        config.get("spectral", "tol", default_tolerance));
    const double growth_tolerance = config.get(
        "spectral", "growth_tol", 1e-8);
    const double residual_limit = residual_tolerance<T>(config);
    const double dt = config.get("ns", "dt", 0.001);

    if (nev <= 0 || max_nev <= 0) {
        throw std::invalid_argument("Fourier block is too small for ARPACK");
    }
    if (nev > max_nev) {
        nev = max_nev;
    }

    for (;;) {
        int ncv = requested_ncv > 0
            ? std::max(requested_ncv, nev+2)
            : std::max(2*nev+2, nev+8);
        ncv = std::min(ncv, n);
        if (ncv-nev < 2) {
            nev = ncv-2;
        }

        result.nev = nev;
        result.ncv = ncv;
        result.arpack_starts = 0;
        result.arpack_lr_starts = 0;
        double best_magnitude = -1;
        vector<complex<T>> best_eigenvalues;

        auto run_starts = [&](typename arpack_solver<T>::WhichEigenvalues which,
                              int count, int start_offset) {
          for (int local_start = 0; local_start < count; ++local_start) {
            const int start = start_offset + local_start;
            arpack_solver<T> solver(n, maxit, arpack_solver<T>::standard, which,
                                    arpack_solver<T>::fixed, tolerance);
            solver.set_ncv(ncv);

            vector<T> residual(n);
            const T seed = static_cast<T>(residual_seed + start);
            for (int i = 0; i < n; ++i) {
              const T x = static_cast<T>(i + 1);
              residual[i] =
                  std::sin((T(0.371) + T(0.017) * seed) * x +
                           T(0.17) * index.m + T(0.131) * seed) +
                  T(0.5) * std::cos((T(0.193) + T(0.011) * seed) * x +
                                    T(0.11) * index.l - T(0.073) * seed);
            }
            solver.set_resid(residual.data());

            vector<complex<T>> eigenvalues;
            vector<vector<T>> eigenvectors;
            int calls = 0;
            double max_leakage = 0;
            solver.solve(
                [&](T *y, const T *x) {
                  block->apply(y, x);
                  max_leakage =
                      std::max(max_leakage, block->last_fourier_leakage());
                  ++calls;
                },
                eigenvalues, eigenvectors, nev);

            result.operator_calls += calls;
            ++result.arpack_starts;
            if (which == arpack_solver<T>::largest_real_part) {
              ++result.arpack_lr_starts;
            }
            if (solver.last_naupd_info() == -8) {
              ++result.arpack_failed_starts;
            }
            result.max_leakage = std::max(result.max_leakage, max_leakage);
            double leading_magnitude = -1;
            for (const auto &value : eigenvalues) {
              leading_magnitude = std::max(
                  leading_magnitude, static_cast<double>(std::abs(value)));
            }
            if (leading_magnitude > best_magnitude) {
              best_magnitude = leading_magnitude;
              result.arpack_info = solver.last_naupd_info();
              result.arpack_iterations = solver.last_iterations();
              result.arpack_nconv = solver.last_nconv();
              best_eigenvalues = std::move(eigenvalues);
            } else if (best_magnitude < 0) {
              result.arpack_info = solver.last_naupd_info();
              result.arpack_iterations = solver.last_iterations();
              result.arpack_nconv = solver.last_nconv();
            }
          }
        };

        auto has_unstable = [&](const vector<complex<T>>& eigenvalues) {
            return std::any_of(eigenvalues.begin(), eigenvalues.end(),
                [&](const complex<T>& value) {
                    const double magnitude = std::abs(value);
                    const double growth = magnitude > 0
                        ? std::log(magnitude)/(operator_steps*dt)
                        : -INFINITY;
                    return growth > growth_tolerance;
                });
        };

        run_starts(arpack_solver<T>::largest_magnitude, probe_starts, 0);
        if (!has_unstable(best_eigenvalues)) {
            // A repeated or strongly nonnormal dominant eigenspace can be
            // missed by an LM restart. LR is an independent screen near the
            // unit circle; any positive hit is verified by the dense solve.
            run_starts(arpack_solver<T>::largest_real_part,
                       probe_lr_starts, probe_starts);
        }
        result.eigenvalues = std::move(best_eigenvalues);

        int unstable = 0;
        int stable = 0;
        for (const auto& value : result.eigenvalues) {
            const double magnitude = std::abs(value);
            const double growth = magnitude > 0
                ? std::log(magnitude)/(operator_steps*dt)
                : -INFINITY;
            if (growth > growth_tolerance) {
                ++unstable;
            } else {
                ++stable;
            }
        }
        result.candidate = unstable > 0;
        result.guard_reached = stable >= stable_guard;

        if (result.guard_reached || nev >= max_nev) {
            break;
        }
        const int next_nev = std::min(max_nev, std::max(nev+2, 2*nev));
        if (next_nev == nev) {
            break;
        }
        nev = next_nev;
    }

    const bool dense_candidates =
        config.get("spectral", "dense_candidates", 1) != 0;
    const bool dense_all = config.get("spectral", "dense_all", 0) != 0;
    if (dense_all
        || (dense_candidates && (result.candidate || !result.guard_reached))) {
        result.dense_spectrum = fdm::solve_ns_cyl_dense_block(
            *block, dt, growth_tolerance, residual_limit);
        result.max_leakage = std::max(
            result.max_leakage,
            result.dense_spectrum.max_fourier_leakage);
        result.dense_computed = true;
    }

    return result;
}

template<typename T>
void run(const Config& config) {
    const int nphi = config.get("ns", "nphi", 32);
    const int nz = config.get("ns", "nz", 32);
    const int m_min = std::max(0, config.get("spectral", "m_min", 0));
    const int m_max = std::min(nphi/2,
        config.get("spectral", "m_max", nphi/2));
    const int l_min = std::max(0, config.get("spectral", "l_min", 0));
    const int l_max = std::min(nz/2,
        config.get("spectral", "l_max", nz/2));
    const bool include_zero = config.get("spectral", "include_zero", 0) != 0;
    const int operator_steps = config.get("spectral", "operator_steps", 1);
    const double dt = config.get("ns", "dt", 0.001);
    const double growth_tolerance = config.get(
        "spectral", "growth_tol", 1e-8);
    const double residual_limit = residual_tolerance<T>(config);
    const double condition_limit = config.get(
        "spectral", "condition_limit", 1e10);

    if (m_min > m_max || l_min > l_max) {
        throw std::invalid_argument("empty Fourier block range");
    }

    vector<BlockIndex> blocks;
    for (int m = m_min; m <= m_max; ++m) {
        for (int l = l_min; l <= l_max; ++l) {
            if (!include_zero && m == 0 && l == 0) {
                continue;
            }
            blocks.push_back({m, l});
        }
    }

    int threads = config.get("spectral", "threads", 1);
#ifdef _OPENMP
    if (threads <= 0) {
        threads = omp_get_max_threads();
    }
    omp_set_max_active_levels(1);
#else
    threads = 1;
#endif

    printf("NSCyl real-packed Fourier ARPACK probe\n");
    printf("grid: nr=%d nz=%d nphi=%d  Re=%.9g dt=%.9g "
           "r=[%.9g,%.9g] z=[%.9g,%.9g]\n",
           config.get("ns", "nr", 32), nz, nphi,
           config.get("ns", "Re", 1.0), dt,
           config.get("ns", "r", M_PI/2),
           config.get("ns", "R", M_PI),
           config.get("ns", "h1", 0.0),
           config.get("ns", "h2", 10.0));
    printf("blocks=%zu m=[%d,%d] l=[%d,%d] threads=%d\n",
           blocks.size(), m_min, m_max, l_min, l_max, threads);
    printf("selection: growth_tol=%.3e residual_tol=%.3e "
           "condition_limit=%.3e\n",
           growth_tolerance, residual_limit, condition_limit);
    printf("packing: q=cosine, N-q=sine; endpoints 0/Nyquist have one phase\n");

    vector<ProbeResult<T>> results(blocks.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
    for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
        try {
            results[i] = probe_block<T>(config, blocks[i]);
        } catch (const std::exception& error) {
            results[i].block = blocks[i];
            results[i].error = error.what();
        }
    }

    int probe_candidate_count = 0;
    int dense_candidate_count = 0;
    int dense_unstable_count = 0;
    int dense_unstable_group_count = 0;
    int dense_rejected_count = 0;
    int dense_block_count = 0;
    fdm::NSCylSpectralModeSet<T> mode_set;
    for (const auto& result : results) {
        if (!result.error.empty()) {
            printf("block (m=%d,l=%d): ERROR: %s\n",
                   result.block.m, result.block.l, result.error.c_str());
            continue;
        }

        printf("block (m=%d,l=%d): D=%d phases=%d arpack_n=%d "
               "nev=%d ncv=%d starts=%d lr_starts=%d failed_starts=%d "
               "calls=%d info=%d "
               "iterations=%d "
               "nconv=%d leakage=%.3e guard=%s%s\n",
               result.block.m, result.block.l,
               result.radial_size, result.phase_count, result.arpack_size,
               result.nev, result.ncv, result.arpack_starts,
               result.arpack_lr_starts,
               result.arpack_failed_starts,
               result.operator_calls,
               result.arpack_info, result.arpack_iterations,
               result.arpack_nconv, result.max_leakage,
               result.guard_reached ? "yes" : "no",
               result.candidate ? " CANDIDATE" : "");

        const auto indices = sorted_indices(result.eigenvalues);
        if (!indices.empty()) {
            const auto leading = result.eigenvalues[indices.front()];
            const double magnitude = std::abs(leading);
            const double growth = magnitude > 0
                ? std::log(magnitude)/(operator_steps*dt)
                : -INFINITY;
            printf("LEADING m=%d l=%d abs=%.16e real=%.16e imag=%+.16e "
                   "growth=%+.9e endpoint=%d\n",
                   result.block.m, result.block.l, magnitude,
                   static_cast<double>(leading.real()),
                   static_cast<double>(leading.imag()), growth,
                   (2*result.block.m == nphi || 2*result.block.l == nz)
                       ? 1 : 0);
        }

        if (result.dense_computed) {
            const auto& spectrum = result.dense_spectrum;
            const auto dense_indices = sorted_indices(spectrum.eigenvalues);
            const auto leading = spectrum.eigenvalues[dense_indices.front()];
            const double leading_magnitude = std::abs(leading);
            const double leading_growth = std::log(leading_magnitude)
                /(operator_steps*dt);
            printf("DENSE_COUNT m=%d l=%d unstable=%d groups=%d "
                   "total=%d calls=%d "
                   "leading_abs=%.16e leading_real=%.16e "
                   "leading_imag=%+.16e leading_growth=%+.9e "
                   "max_right_res=%.3e max_left_res=%.3e\n",
                   result.block.m, result.block.l,
                   spectrum.filterable_unstable_dimension(),
                   spectrum.filterable_unstable_group_count(),
                   static_cast<int>(spectrum.eigenvalues.size()),
                   spectrum.operator_calls,
                   leading_magnitude,
                   static_cast<double>(leading.real()),
                   static_cast<double>(leading.imag()), leading_growth,
                   spectrum.max_right_residual,
                   spectrum.max_left_residual);

            vector<int> mode_indices(spectrum.modes.size());
            std::iota(mode_indices.begin(), mode_indices.end(), 0);
            std::stable_sort(mode_indices.begin(), mode_indices.end(),
                [&](int a, int b) {
                    return spectrum.modes[a].growth_rate
                        > spectrum.modes[b].growth_rate;
                });
            int unstable_position = 0;
            for (int mode_index : mode_indices) {
                const auto& mode = spectrum.modes[mode_index];
                if (!mode.growing) {
                    continue;
                }
                printf("%s m=%d l=%d index=%d columns=%d abs=%.16e "
                       "real=%.16e imag=%+.16e growth=%+.9e "
                       "frequency=%+.9e right_res=%.3e left_res=%.3e\n",
                       mode.residual_accepted
                           ? "DENSE_UNSTABLE" : "DENSE_REJECTED",
                       result.block.m, result.block.l, unstable_position++,
                       mode.column_count, std::abs(mode.multiplier),
                       static_cast<double>(mode.multiplier.real()),
                       static_cast<double>(mode.multiplier.imag()),
                       mode.growth_rate, mode.frequency,
                       mode.right_residual, mode.left_residual);
            }
        } else if (result.guard_reached) {
            printf("DENSE_COUNT m=%d l=%d unstable=0 groups=0 total=0 calls=0 "
                   "leading_abs=nan leading_growth=nan screened_by_probe=1\n",
                   result.block.m, result.block.l);
        }
        for (int position = 0; position < static_cast<int>(indices.size());
             ++position) {
            const auto value = result.eigenvalues[indices[position]];
            const double magnitude = std::abs(value);
            const double growth = magnitude > 0
                ? std::log(magnitude)/(operator_steps*dt)
                : -INFINITY;
            printf("  %3d |mu|=%.16e mu=(%.16e,%+.16e) growth=%+.9e\n",
                   position, magnitude,
                   static_cast<double>(value.real()),
                   static_cast<double>(value.imag()), growth);
        }

        probe_candidate_count += result.candidate ? 1 : 0;
        if (result.dense_computed) {
            const auto& spectrum = result.dense_spectrum;
            ++dense_block_count;
            dense_candidate_count +=
                spectrum.filterable_unstable_dimension() > 0 ? 1 : 0;
            dense_unstable_count += spectrum.filterable_unstable_dimension();
            dense_unstable_group_count +=
                spectrum.filterable_unstable_group_count();
            dense_rejected_count += spectrum.growing_dimension()
                -spectrum.filterable_unstable_dimension();
            mode_set.append_filterable(spectrum);
        }
    }

    mode_set.sort_by_block_and_growth();
    fflush(stdout);
    const fdm::NSCylSpectralProjector<T> projector(
        mode_set, condition_limit);
    for (const auto& block : projector.blocks()) {
        printf("GRAM m=%d l=%d dimension=%d condition=%.9e "
               "gram_cond_inf=%.9e min_pivot=%.9e\n",
               block.m(), block.l(), block.dimension(),
               block.condition_number(), block.gram_condition_number(),
               block.min_pivot());
    }
    const string output = config.get("spectral", "output", string());
    if (!output.empty()) {
        const auto metadata = fdm::make_ns_cyl_spectral_metadata<T>(config);
        fdm::NSCylSpectralStorage(output).save(mode_set, metadata);
        printf("saved spectrum: %s groups=%zu real_dimension=%d\n",
               output.c_str(), mode_set.size(), mode_set.real_dimension());
    }

    printf("probe candidate blocks: %d / %zu\n",
           probe_candidate_count, results.size());
    if (dense_block_count > 0) {
        printf("dense unstable blocks: %d / %d computed (%zu scanned)\n",
               dense_candidate_count, dense_block_count, results.size());
        printf("filterable unstable modes: groups=%d real_columns=%d "
               "rejected_columns=%d\n",
               dense_unstable_group_count, dense_unstable_count,
               dense_rejected_count);
        printf("mode set: groups=%zu real_dimension=%d\n",
               mode_set.size(), mode_set.real_dimension());
        printf("projector: blocks=%zu real_dimension=%d\n",
               projector.blocks().size(), projector.real_dimension());
    }
    printf("note: complex pairs are stored as adjacent real Re(v), Im(v) "
           "columns\n");
}

} // namespace

int main(int argc, char** argv) {
    string config_name = "ns_cyl_spectral_probe.ini";
    for (int i = 1; i+1 < argc; ++i) {
        if (!strcmp(argv[i], "-c")) {
            config_name = argv[i+1];
        }
    }

    Config config;
    config.open(config_name);
    config.rewrite(argc, argv);

    try {
        const string datatype = config.get("solver", "datatype", "double");
        if (datatype == "float") {
            run<float>(config);
        } else {
            run<double>(config);
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "spectral probe failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
