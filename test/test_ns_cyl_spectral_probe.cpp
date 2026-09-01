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
#include "blas.h"
#include "config.h"
#include "ns_cyl_fourier_block.h"

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
    double max_leakage = 0;
    bool candidate = false;
    bool guard_reached = false;
    bool dense_computed = false;
    int dense_operator_calls = 0;
    int dense_unstable_count = 0;
    string error;
    double dense_max_right_residual = 0;
    double dense_max_left_residual = 0;
    vector<complex<T>> eigenvalues;
    vector<complex<T>> dense_eigenvalues;
    vector<double> dense_right_residual;
    vector<double> dense_left_residual;
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

// matrix - column major, разрушается. Для комплексно сопряженной пары i, i+1
// собственный вектор это столбец i (Re) и столбец i+1 (Im).
template<typename T>
int dense_geev(int n, T* matrix, T* real, T* imaginary, T* left, T* right) {
    vector<T> work(8*n);
    int info = 0;
    fdm::lapack::geev("V", "V", n, matrix, n, real, imaginary,
                      left, n, right, n, work.data(), 8*n, &info);
    return info;
}

// y = A x, a - column major
template<typename T>
void matvec(int n, const T* a, const T* x, T* y) {
    for (int row = 0; row < n; ++row) {
        y[row] = 0;
    }
    for (int column = 0; column < n; ++column) {
        const T value = x[column];
        for (int row = 0; row < n; ++row) {
            y[row] += a[static_cast<std::size_t>(column)*n+row]*value;
        }
    }
}

// y = A^t x
template<typename T>
void matvec_transposed(int n, const T* a, const T* x, T* y) {
    for (int row = 0; row < n; ++row) {
        T sum = 0;
        for (int column = 0; column < n; ++column) {
            sum += a[static_cast<std::size_t>(row)*n+column]*x[column];
        }
        y[row] = sum;
    }
}

template<typename T>
double norm2(int n, const T* x) {
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i])*x[i];
    }
    return sum;
}

// Невязки считаются независимо от lapack:
//   right = ||A r - mu r|| / (||A|| ||r||)
//   left  = ||A^t l - conj(mu) l|| / (||A|| ||l||)
// Вещественное mu -- один столбец, сопряженная пара -- два соседних, и оба
// вектора пары дают одну и ту же невязку.
template<typename T>
void eigen_residuals(int n, const T* a, const T* real, const T* imaginary,
                     const T* left, const T* right, double matrix_norm,
                     vector<double>& right_residual,
                     vector<double>& left_residual)
{
    right_residual.assign(n, 0);
    left_residual.assign(n, 0);

    vector<T> ar(n);
    vector<T> ai(n);

    for (int i = 0; i < n; ) {
        const int count = (imaginary[i] == T(0)) ? 1 : 2;
        const T* vr = right+static_cast<std::size_t>(i)*n;
        const T* vl = left+static_cast<std::size_t>(i)*n;
        const T wr = real[i];
        const T wi = imaginary[i];

        double residual = 0;
        double scale = 0;

        if (count == 1) {
            matvec(n, a, vr, ar.data());
            for (int k = 0; k < n; ++k) {
                ar[k] -= wr*vr[k];
            }
            residual = norm2(n, ar.data());
            scale = norm2(n, vr);
        } else {
            const T* vr2 = vr+n;
            matvec(n, a, vr, ar.data());
            matvec(n, a, vr2, ai.data());
            for (int k = 0; k < n; ++k) {
                ar[k] -= wr*vr[k]-wi*vr2[k];
                ai[k] -= wr*vr2[k]+wi*vr[k];
            }
            residual = norm2(n, ar.data())+norm2(n, ai.data());
            scale = norm2(n, vr)+norm2(n, vr2);
        }
        const double value = std::sqrt(residual)/(matrix_norm*std::sqrt(scale));
        for (int k = 0; k < count; ++k) {
            right_residual[i+k] = value;
        }

        if (count == 1) {
            matvec_transposed(n, a, vl, ar.data());
            for (int k = 0; k < n; ++k) {
                ar[k] -= wr*vl[k];
            }
            residual = norm2(n, ar.data());
            scale = norm2(n, vl);
        } else {
            const T* vl2 = vl+n;
            matvec_transposed(n, a, vl, ar.data());
            matvec_transposed(n, a, vl2, ai.data());
            for (int k = 0; k < n; ++k) {
                ar[k] -= wr*vl[k]+wi*vl2[k];
                ai[k] -= wr*vl2[k]-wi*vl[k];
            }
            residual = norm2(n, ar.data())+norm2(n, ai.data());
            scale = norm2(n, vl)+norm2(n, vl2);
        }
        const double left_value =
            std::sqrt(residual)/(matrix_norm*std::sqrt(scale));
        for (int k = 0; k < count; ++k) {
            left_residual[i+k] = left_value;
        }

        i += count;
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
    const double default_tolerance = std::max(
        1e-8, 8.0*static_cast<double>(std::numeric_limits<T>::epsilon()));
    const T tolerance = static_cast<T>(
        config.get("spectral", "tol", default_tolerance));
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
            solver.solve([&](T* y, const T* x) {
                block->apply(y, x);
                max_leakage = std::max(max_leakage,
                                       block->last_fourier_leakage());
                ++calls;
            }, eigenvalues, eigenvectors, nev);

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
        vector<T> left(static_cast<std::size_t>(n)*n);
        vector<T> rightv(static_cast<std::size_t>(n)*n);
        vector<T> factored(matrix);   // geev разрушает матрицу
        const int info = dense_geev(n, factored.data(), real.data(),
                                    imaginary.data(), left.data(),
                                    rightv.data());
        if (info != 0) {
            throw std::runtime_error(
                "geev failed with info="+std::to_string(info));
        }

        const double matrix_norm = std::sqrt(
            norm2(static_cast<int>(matrix.size()), matrix.data()));
        eigen_residuals(n, matrix.data(), real.data(), imaginary.data(),
                        left.data(), rightv.data(), matrix_norm,
                        result.dense_right_residual,
                        result.dense_left_residual);
        for (int i = 0; i < n; ++i) {
            result.dense_max_right_residual = std::max(
                result.dense_max_right_residual, result.dense_right_residual[i]);
            result.dense_max_left_residual = std::max(
                result.dense_max_left_residual, result.dense_left_residual[i]);
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
                   "leading_imag=%+.16e leading_growth=%+.9e "
                   "max_right_res=%.3e max_left_res=%.3e\n",
                   result.block.m, result.block.l,
                   result.dense_unstable_count,
                   static_cast<int>(result.dense_eigenvalues.size()),
                   result.dense_operator_calls,
                   leading_magnitude,
                   static_cast<double>(leading.real()),
                   static_cast<double>(leading.imag()), leading_growth,
                   result.dense_max_right_residual,
                   result.dense_max_left_residual);

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
                       "real=%.16e imag=%+.16e growth=%+.9e "
                       "right_res=%.3e left_res=%.3e\n",
                       result.block.m, result.block.l, unstable_position++,
                       magnitude, static_cast<double>(value.real()),
                       static_cast<double>(value.imag()), growth,
                       result.dense_right_residual[dense_index],
                       result.dense_left_residual[dense_index]);
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
    string config_name = "ns_cyl_spectral_probe.ini";
    for (int i = 1; i+1 < argc; ++i) {
        if (!strcmp(argv[i], "-c")) {
            config_name = argv[i+1];
        }
    }

    Config config;
    config.open(config_name);
    config.rewrite(argc, argv);

    const string datatype = config.get("solver", "datatype", "double");
    if (datatype == "float") {
        run<float>(config);
    } else {
        run<double>(config);
    }
    return 0;
}
