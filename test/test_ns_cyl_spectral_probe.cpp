#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <exception>
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

using fdm::arpack_solver;
using fdm::NSCylFourierBlockReference;
using std::complex;
using std::string;
using std::vector;

extern "C" int LAPACKE_dgeev(
    int matrix_layout, char jobvl, char jobvr, int n, double* a, int lda,
    double* wr, double* wi, double* vl, int ldvl, double* vr, int ldvr);
extern "C" int LAPACKE_sgeev(
    int matrix_layout, char jobvl, char jobvr, int n, float* a, int lda,
    float* wr, float* wi, float* vl, int ldvl, float* vr, int ldvr);

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
    double max_leakage = 0;
    bool candidate = false;
    bool guard_reached = false;
    bool dense_computed = false;
    int dense_operator_calls = 0;
    int dense_unstable_count = 0;
    string error;
    vector<complex<T>> eigenvalues;
    vector<complex<T>> dense_eigenvalues;
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
int dense_geev(int n, T* matrix, T* real, T* imaginary) {
    constexpr int lapack_column_major = 102;
    T unused_left = 0;
    T unused_right = 0;
    if constexpr (std::is_same_v<T, double>) {
        return LAPACKE_dgeev(lapack_column_major, 'N', 'N', n,
                             matrix, n, real, imaginary,
                             &unused_left, 1, &unused_right, 1);
    } else {
        return LAPACKE_sgeev(lapack_column_major, 'N', 'N', n,
                             matrix, n, real, imaginary,
                             &unused_left, 1, &unused_right, 1);
    }
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
    const T tolerance = static_cast<T>(
        config.get("spectral", "tol", 1e-8));
    const double growth_tolerance = config.get(
        "spectral", "growth_tol", 1e-8);
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
        result.arpack_starts = probe_starts;
        double best_magnitude = -1;

        for (int start = 0; start < probe_starts; ++start) {
            arpack_solver<T> solver(
                n, maxit,
                arpack_solver<T>::standard,
                arpack_solver<T>::largest_magnitude,
                arpack_solver<T>::fixed,
                tolerance);
            solver.set_ncv(ncv);

            vector<T> residual(n);
            const T seed = static_cast<T>(residual_seed+start);
            for (int i = 0; i < n; ++i) {
                const T x = static_cast<T>(i+1);
                residual[i] =
                    std::sin((T(0.371)+T(0.017)*seed)*x
                             +T(0.17)*index.m+T(0.131)*seed)
                    +T(0.5)*std::cos((T(0.193)+T(0.011)*seed)*x
                                     +T(0.11)*index.l-T(0.073)*seed);
            }
            solver.set_resid(residual.data());

            vector<complex<T>> eigenvalues;
            vector<vector<T>> eigenvectors;
            int calls = 0;
            double max_leakage = 0;
            // The bundled f2c ARPACK keeps static routine state. Its calls
            // are serialized; dense verification may still run per block.
#ifdef _OPENMP
#pragma omp critical(fdm_ns_cyl_probe_arpack_call)
#endif
            {
                solver.solve([&](T* y, const T* x) {
                    block->apply(y, x);
                    max_leakage = std::max(max_leakage,
                                           block->last_fourier_leakage());
                    ++calls;
                }, eigenvalues, eigenvectors, nev);
            }

            result.operator_calls += calls;
            result.max_leakage = std::max(result.max_leakage, max_leakage);
            double leading_magnitude = -1;
            for (const auto& value : eigenvalues) {
                leading_magnitude = std::max(
                    leading_magnitude, static_cast<double>(std::abs(value)));
            }
            if (leading_magnitude > best_magnitude) {
                best_magnitude = leading_magnitude;
                result.arpack_info = solver.last_naupd_info();
                result.arpack_iterations = solver.last_iterations();
                result.arpack_nconv = solver.last_nconv();
                result.eigenvalues = std::move(eigenvalues);
            }
        }

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
    if (dense_all || (dense_candidates && result.candidate)) {
        vector<T> matrix(static_cast<std::size_t>(n)*n);
        vector<T> basis(n, T(0));
        vector<T> image(n);
        for (int column = 0; column < n; ++column) {
            basis[column] = T(1);
            block->apply(image.data(), basis.data());
            basis[column] = T(0);
            for (int row = 0; row < n; ++row) {
                matrix[static_cast<std::size_t>(column)*n+row] = image[row];
            }
        }
        result.dense_operator_calls = n;
        result.max_leakage = std::max(result.max_leakage,
                                      block->last_fourier_leakage());

        vector<T> real(n);
        vector<T> imaginary(n);
        const int info = dense_geev(n, matrix.data(), real.data(),
                                    imaginary.data());
        if (info != 0) {
            throw std::runtime_error(
                "LAPACKE geev failed with info="+std::to_string(info));
        }

        result.dense_computed = true;
        result.dense_eigenvalues.reserve(n);
        const double duration = operator_steps*dt;
        const double unstable_threshold =
            std::exp(growth_tolerance*duration);
        for (int i = 0; i < n; ++i) {
            const complex<T> value(real[i], imaginary[i]);
            result.dense_eigenvalues.push_back(value);
            if (std::abs(value) > unstable_threshold) {
                ++result.dense_unstable_count;
            }
        }
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
    int dense_block_count = 0;
    for (const auto& result : results) {
        if (!result.error.empty()) {
            printf("block (m=%d,l=%d): ERROR: %s\n",
                   result.block.m, result.block.l, result.error.c_str());
            continue;
        }

        printf("block (m=%d,l=%d): D=%d phases=%d arpack_n=%d "
               "nev=%d ncv=%d starts=%d calls=%d info=%d iterations=%d "
               "nconv=%d leakage=%.3e guard=%s%s\n",
               result.block.m, result.block.l,
               result.radial_size, result.phase_count, result.arpack_size,
               result.nev, result.ncv, result.arpack_starts,
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
            const auto dense_indices = sorted_indices(result.dense_eigenvalues);
            const auto leading = result.dense_eigenvalues[dense_indices.front()];
            const double leading_magnitude = std::abs(leading);
            const double leading_growth = std::log(leading_magnitude)
                /(operator_steps*dt);
            printf("DENSE_COUNT m=%d l=%d unstable=%d total=%d calls=%d "
                   "leading_abs=%.16e leading_real=%.16e "
                   "leading_imag=%+.16e leading_growth=%+.9e\n",
                   result.block.m, result.block.l,
                   result.dense_unstable_count,
                   static_cast<int>(result.dense_eigenvalues.size()),
                   result.dense_operator_calls,
                   leading_magnitude,
                   static_cast<double>(leading.real()),
                   static_cast<double>(leading.imag()), leading_growth);

            int unstable_position = 0;
            for (int dense_index : dense_indices) {
                const auto value = result.dense_eigenvalues[dense_index];
                const double magnitude = std::abs(value);
                const double growth = magnitude > 0
                    ? std::log(magnitude)/(operator_steps*dt)
                    : -INFINITY;
                if (growth <= config.get("spectral", "growth_tol", 1e-8)) {
                    continue;
                }
                printf("DENSE_UNSTABLE m=%d l=%d index=%d abs=%.16e "
                       "real=%.16e imag=%+.16e growth=%+.9e\n",
                       result.block.m, result.block.l, unstable_position++,
                       magnitude, static_cast<double>(value.real()),
                       static_cast<double>(value.imag()), growth);
            }
        } else if (result.guard_reached) {
            printf("DENSE_COUNT m=%d l=%d unstable=0 total=0 calls=0 "
                   "leading_abs=nan leading_growth=nan certified_by_probe=1\n",
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
            ++dense_block_count;
            dense_candidate_count += result.dense_unstable_count > 0 ? 1 : 0;
            dense_unstable_count += result.dense_unstable_count;
        }
    }

    printf("probe candidate blocks: %d / %zu\n",
           probe_candidate_count, results.size());
    if (dense_block_count > 0) {
        printf("dense unstable blocks: %d / %d computed (%zu scanned)\n",
               dense_candidate_count, dense_block_count, results.size());
        printf("dense unstable eigenvalues in real packed blocks: %d\n",
               dense_unstable_count);
    }
    printf("note: for complex Ritz pairs dneupd stores Re(v), Im(v) in "
           "adjacent real columns\n");
}

} // namespace

int main(int argc, char** argv) {
    Config config;
    config.open("ns_rect.ini");
    config.rewrite(argc, argv);

    const string datatype = config.get("solver", "datatype", "double");
    verify(datatype == "double",
           "fdm_ns_cyl_spectral_probe currently requires double precision");
    run<double>(config);
    return 0;
}
