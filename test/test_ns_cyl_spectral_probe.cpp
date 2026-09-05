#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef FDM_HAVE_SYCL
#include "ns_cyl_fourier_batch_sycl.h"
#include "ns_cyl_fourier_block_sycl.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include "arpack_solver.h"
#include "config.h"
#include "ns_cyl_fourier_batch.h"
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

template<typename T, typename Block>
ProbeResult<T> probe_block_impl(
    const Config& config, BlockIndex index, Block& block,
    bool compute_dense=true)
{
    ProbeResult<T> result;
    result.block = index;

    const int operator_steps = config.get("spectral", "operator_steps", 1);
    result.radial_size = block.radial_size();
    result.phase_count = block.phase_count();
    result.arpack_size = block.size();

    const int n = block.size();
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
                  block.apply(y, x);
                  max_leakage =
                      std::max(max_leakage, block.last_fourier_leakage());
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
    if (compute_dense && (dense_all
        || (dense_candidates && (result.candidate || !result.guard_reached)))) {
        result.dense_spectrum = fdm::solve_ns_cyl_dense_block(
            block, dt, growth_tolerance, residual_limit);
        result.max_leakage = std::max(
            result.max_leakage,
            result.dense_spectrum.max_fourier_leakage);
        result.dense_computed = true;
    }

    return result;
}

template<typename T>
ProbeResult<T> probe_cpu_block(const Config& config, BlockIndex index) {
    using Block = NSCylFourierBlockReference<T, false>;
    const int operator_steps = config.get("spectral", "operator_steps", 1);
    std::unique_ptr<Block> block;

    // FFTW plan creation is serialized; executing distinct plans is safe.
#ifdef _OPENMP
#pragma omp critical(fdm_ns_cyl_probe_fft_planning)
#endif
    {
        block = std::make_unique<Block>(
            config, index.m, index.l, operator_steps);
    }
    return probe_block_impl<T>(config, index, *block);
}

#ifdef FDM_HAVE_SYCL
sycl::device select_sycl_device() {
    for (const auto& platform : sycl::platform::get_platforms()) {
        for (const auto& device : platform.get_devices()) {
            if (device.is_gpu()) {
                return device;
            }
        }
    }
    return sycl::device{sycl::cpu_selector_v};
}

ProbeResult<float> probe_sycl_block(
    sycl::queue& queue, const Config& config, BlockIndex index)
{
    const int operator_steps = config.get("spectral", "operator_steps", 1);
    fdm::NSCylSyclFourierBlockReference<float> block(
        queue, config, index.m, index.l, operator_steps);
    return probe_block_impl<float>(config, index, block);
}
#endif

template<typename T, typename BatchOperator>
class BatchedArpackCoordinator {
public:
    using Request = fdm::NSCylFourierBatchRequest<T>;

    BatchedArpackCoordinator(BatchOperator& op, int slots)
        : op_(op), states_(slots), active_(slots)
    { }

    void apply(std::size_t slot, BlockIndex block,
               T* output, const T* input, int size) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (failure_) {
            std::rethrow_exception(failure_);
        }
        State& state = states_.at(slot);
        if (state.waiting) {
            throw std::logic_error("duplicate outstanding batched matvec");
        }
        state.block = block;
        state.input = input;
        state.output = output;
        state.size = size;
        state.waiting = true;
        ++pending_;
        changed_.notify_all();
        changed_.wait(lock, [&] { return !state.waiting || failure_; });
        if (failure_) {
            std::rethrow_exception(failure_);
        }
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        --active_;
        changed_.notify_all();
    }

    int drive(int progress_interval) {
        int batches = 0;
        for (;;) {
            vector<Request> requests;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock, [&] {
                    return failure_ || active_ == 0 || pending_ == active_;
                });
                if (failure_) {
                    break;
                }
                if (active_ == 0) {
                    break;
                }
                requests.reserve(pending_);
                for (std::size_t i = 0; i < states_.size(); ++i) {
                    const State& state = states_[i];
                    if (state.waiting) {
                        requests.push_back({
                            state.block.m, state.block.l,
                            state.input, state.output, state.size});
                    }
                }
            }

            try {
                op_.apply(requests);
                ++batches;
                if (progress_interval > 0
                    && batches % progress_interval == 0) {
                    printf("BATCHED_ARPACK_PROGRESS physical_batches=%d\n",
                           batches);
                    fflush(stdout);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                failure_ = std::current_exception();
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (State& state : states_) {
                    state.waiting = false;
                }
                pending_ = 0;
                changed_.notify_all();
            }
        }
        return batches;
    }

private:
    struct State {
        BlockIndex block{};
        const T* input = nullptr;
        T* output = nullptr;
        int size = 0;
        bool waiting = false;
    };

    BatchOperator& op_;
    vector<State> states_;
    std::mutex mutex_;
    std::condition_variable changed_;
    int active_ = 0;
    int pending_ = 0;
    std::exception_ptr failure_;
};

template<typename T, typename Coordinator>
class BatchedBlockProxy {
public:
    using value_type = T;

    BatchedBlockProxy(Coordinator& coordinator, std::size_t coordinator_index,
                      BlockIndex block, int nr, int nz, int nphi,
                      int operator_steps)
        : coordinator_(coordinator)
        , coordinator_index_(coordinator_index)
        , block_(block)
        , radial_size_(4*nr-1)
        , phase_count_(phase_count(block.m, nphi)*phase_count(block.l, nz))
        , size_(radial_size_*phase_count_
                -(block.m == 0 && block.l == 0 ? 1 : 0))
        , operator_steps_(operator_steps)
    { }

    int radial_size() const { return radial_size_; }
    int phase_count() const { return phase_count_; }
    int size() const { return size_; }
    int operator_steps() const { return operator_steps_; }
    int m() const { return block_.m; }
    int l() const { return block_.l; }
    bool pressure_gauge_fixed() const {
        return block_.m == 0 && block_.l == 0;
    }
    double last_fourier_leakage() const { return 0; }

    void apply(T* output, const T* input) {
        coordinator_.apply(
            coordinator_index_, block_, output, input, size_);
    }

private:
    Coordinator& coordinator_;
    std::size_t coordinator_index_;
    BlockIndex block_;
    int radial_size_;
    int phase_count_;
    int size_;
    int operator_steps_;

    static int phase_count(int q, int n) {
        return q == 0 || 2*q == n ? 1 : 2;
    }
};

template<typename T>
void set_dense_metadata(ProbeResult<T>& probe, vector<T>& matrix,
                        int operator_steps, double dt,
                        double growth_tolerance, double residual_limit) {
    auto spectrum = fdm::analyze_ns_cyl_dense_matrix(
        matrix.data(), probe.arpack_size, operator_steps*dt,
        growth_tolerance, residual_limit);
    spectrum.m = probe.block.m;
    spectrum.l = probe.block.l;
    spectrum.phase_count = probe.phase_count;
    spectrum.radial_size = probe.radial_size;
    spectrum.operator_steps = operator_steps;
    spectrum.operator_calls = probe.arpack_size;
    spectrum.pressure_gauge_fixed =
        probe.block.m == 0 && probe.block.l == 0;
    for (auto& mode : spectrum.modes) {
        mode.m = spectrum.m;
        mode.l = spectrum.l;
        mode.phase_count = spectrum.phase_count;
        mode.radial_size = spectrum.radial_size;
        mode.pressure_gauge_fixed = spectrum.pressure_gauge_fixed;
    }
    probe.dense_spectrum = std::move(spectrum);
    probe.dense_computed = true;
}

template<typename T, typename BatchOperator>
void probe_batched_blocks_impl(const Config& config,
                               const vector<BlockIndex>& blocks,
                               vector<ProbeResult<T>>& results,
                               BatchOperator& op) {
    const int operator_steps = config.get("spectral", "operator_steps", 1);
    const int nr = config.get("ns", "nr", 32);
    const int nz = config.get("ns", "nz", 32);
    const int nphi = config.get("ns", "nphi", 32);
    const int common_batch = config.get("spectral", "batch_blocks", 0);
    int arpack_batch = config.get(
        "spectral", "batch_arnoldi_blocks", common_batch);
    int dense_batch = config.get(
        "spectral", "batch_dense_blocks", common_batch);
    if (arpack_batch <= 0) {
        arpack_batch = std::is_same_v<T, float>
            ? std::min(16, static_cast<int>(blocks.size()))
            : std::min(64, static_cast<int>(blocks.size()));
    }
    if (dense_batch <= 0) {
        dense_batch = std::is_same_v<T, float>
            ? std::min(2, static_cast<int>(blocks.size()))
            : static_cast<int>(blocks.size());
    }
    arpack_batch = std::max(1, std::min(
        arpack_batch, static_cast<int>(blocks.size())));
    dense_batch = std::max(1, std::min(
        dense_batch, static_cast<int>(blocks.size())));
    printf("BATCHED_LAYOUT arnoldi_blocks=%d dense_blocks=%d\n",
           arpack_batch, dense_batch);

    const int worker_count = std::min(
        arpack_batch, static_cast<int>(blocks.size()));
    const int progress_interval = config.get(
        "spectral", "batch_progress_batches", 500);
    const auto arpack_before = std::chrono::steady_clock::now();
    BatchedArpackCoordinator<T, BatchOperator> coordinator(op, worker_count);
    std::atomic<int> next_block{0};
    vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int slot = 0; slot < worker_count; ++slot) {
        workers.emplace_back([&, slot] {
            for (;;) {
                const int result_index = next_block.fetch_add(
                    1, std::memory_order_relaxed);
                if (result_index >= static_cast<int>(blocks.size())) {
                    break;
                }
                try {
                    BatchedBlockProxy<T, decltype(coordinator)> block(
                        coordinator, slot, blocks[result_index],
                        nr, nz, nphi,
                        operator_steps);
                    results[result_index] = probe_block_impl<T>(
                        config, blocks[result_index], block, false);
                } catch (const std::exception& error) {
                    results[result_index].block = blocks[result_index];
                    results[result_index].error = error.what();
                }
            }
            coordinator.finish();
        });
    }
    const int arpack_batches = coordinator.drive(progress_interval);
    for (std::thread& worker : workers) {
        worker.join();
    }

    long long arpack_logical_calls = 0;
    int slowest_calls = -1;
    BlockIndex slowest_block{};
    for (const auto& result : results) {
        arpack_logical_calls += result.operator_calls;
        if (result.operator_calls > slowest_calls) {
            slowest_calls = result.operator_calls;
            slowest_block = result.block;
        }
    }
    const double arpack_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-arpack_before).count();
    const double arpack_fill = arpack_batches > 0
        ? static_cast<double>(arpack_logical_calls)/arpack_batches : 0;
    printf("BATCHED_ARPACK physical_batches=%d logical_calls=%lld "
           "mean_fill=%.2f slowest=(%d,%d):%d seconds=%.3f\n",
           arpack_batches, arpack_logical_calls, arpack_fill,
           slowest_block.m, slowest_block.l, slowest_calls, arpack_seconds);

    const bool dense_candidates =
        config.get("spectral", "dense_candidates", 1) != 0;
    const bool dense_all = config.get("spectral", "dense_all", 0) != 0;
    vector<int> dense_indices;
    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        const auto& result = results[i];
        if (result.error.empty()
            && (dense_all || (dense_candidates
                && (result.candidate || !result.guard_reached)))) {
            dense_indices.push_back(i);
        }
    }

    struct DenseWork {
        vector<T> matrix;
        vector<T> basis;
        vector<T> image;
    };
    vector<DenseWork> work(dense_indices.size());
    int maximum_size = 0;
    for (std::size_t i = 0; i < dense_indices.size(); ++i) {
        const int n = results[dense_indices[i]].arpack_size;
        maximum_size = std::max(maximum_size, n);
        work[i].matrix.resize(static_cast<std::size_t>(n)*n);
        work[i].basis.assign(n, T(0));
        work[i].image.resize(n);
    }

    int dense_batches = 0;
    const auto dense_before = std::chrono::steady_clock::now();
    for (int column = 0; column < maximum_size; ++column) {
        for (int begin = 0; begin < static_cast<int>(dense_indices.size());
             begin += dense_batch) {
            const int end = std::min(
                static_cast<int>(dense_indices.size()), begin+dense_batch);
            vector<fdm::NSCylFourierBatchRequest<T>> requests;
            vector<int> request_work_indices;
            for (int i = begin; i < end; ++i) {
                const int result_index = dense_indices[i];
                const int n = results[result_index].arpack_size;
                if (column >= n) {
                    continue;
                }
                work[i].basis[column] = T(1);
                requests.push_back({
                    results[result_index].block.m,
                    results[result_index].block.l,
                    work[i].basis.data(), work[i].image.data(), n});
                request_work_indices.push_back(i);
            }
            if (!requests.empty()) {
                op.apply(requests);
                ++dense_batches;
            }
            for (int i : request_work_indices) {
                const int n = results[dense_indices[i]].arpack_size;
                work[i].basis[column] = T(0);
                std::copy(work[i].image.begin(), work[i].image.end(),
                          work[i].matrix.begin()+static_cast<std::size_t>(column)*n);
            }
        }
    }
    const double materialize_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-dense_before).count();
    printf("BATCHED_DENSE physical_batches=%d blocks=%zu columns=%d "
           "seconds=%.3f\n", dense_batches, dense_indices.size(),
           maximum_size, materialize_seconds);

    const double dt = config.get("ns", "dt", 0.001);
    const double growth_tolerance = config.get(
        "spectral", "growth_tol", 1e-8);
    const double residual_limit = residual_tolerance<T>(config);
    int threads = config.get("spectral", "threads", 1);
#ifdef _OPENMP
    if (threads <= 0) {
        threads = omp_get_max_threads();
    }
#else
    threads = 1;
#endif
    threads = std::max(1, std::min(
        threads, static_cast<int>(dense_indices.size())));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
    for (int i = 0; i < static_cast<int>(dense_indices.size()); ++i) {
        try {
            set_dense_metadata(
                results[dense_indices[i]], work[i].matrix, operator_steps, dt,
                growth_tolerance, residual_limit);
        } catch (const std::exception& error) {
            results[dense_indices[i]].error = error.what();
        }
    }
    const double dense_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-dense_before).count();
    printf("BATCHED_GEEV threads=%d seconds=%.3f\n",
           threads, dense_seconds-materialize_seconds);
}

template<typename T>
void probe_batched_blocks(const Config& config,
                          const vector<BlockIndex>& blocks,
                          vector<ProbeResult<T>>& results,
                          const string& backend) {
    const int operator_steps = config.get("spectral", "operator_steps", 1);
    if (backend == "cpu") {
        fdm::NSCylFourierBlockBatchReference<T> op(config, operator_steps);
        probe_batched_blocks_impl(config, blocks, results, op);
        return;
    }
#ifdef FDM_HAVE_SYCL
    if constexpr (std::is_same_v<T, float>) {
        const sycl::device device = select_sycl_device();
        printf("SYCL device: %s\n",
               device.get_info<sycl::info::device::name>().c_str());
        sycl::queue queue{device, sycl::property::queue::in_order{}};
        fdm::NSCylSyclFourierBlockBatchReference<T> op(
            queue, config, operator_steps);
        probe_batched_blocks_impl(config, blocks, results, op);
    } else {
        throw std::invalid_argument(
            "SYCL batched probe currently supports datatype=float only");
    }
#else
    (void)config;
    (void)blocks;
    (void)results;
    throw std::runtime_error("spectral probe was built without SYCL");
#endif
}

template<typename T>
class GlobalGaugeLayout {
public:
    GlobalGaugeLayout(int nr, int nz, int nphi, double r0, double dr)
        : layout_(nr, nz, nphi)
        , householder_(layout_.p_size)
    {
        // The gauge condition is w^T p=0 with cell-radius weights w.  This
        // reflector maps the last pressure coordinate to normalized w, so
        // its other columns give a well-conditioned orthonormal gauge basis.
        long double weight_norm2 = 0;
        for (int index = 0; index < layout_.p_size; ++index) {
            const int j = index%layout_.nr+1;
            const long double weight = r0+(j-0.5L)*dr;
            householder_[index] = -weight;
            weight_norm2 += weight*weight;
        }
        const long double weight_norm = std::sqrt(weight_norm2);
        for (long double& value : householder_) {
            value /= weight_norm;
        }
        householder_.back() += 1;
        long double reflector_norm2 = 0;
        for (long double value : householder_) {
            reflector_norm2 += value*value;
        }
        const long double reflector_norm = std::sqrt(reflector_norm2);
        for (long double& value : householder_) {
            value /= reflector_norm;
        }
    }

    int size() const { return layout_.state_size-1; }
    int full_size() const { return layout_.state_size; }

    const fdm::NSCylStateLayout<T>& state_layout() const { return layout_; }

    template<typename Geometry>
    void expand(const Geometry& geometry, const T* reduced, T* full) const {
        (void)geometry;
        std::copy(reduced, reduced+size(), full);
        full[size()] = T(0);
        reflect_pressure(full);
    }

    template<typename Geometry>
    void reduce(const Geometry& geometry, T* full, T* reduced) const {
        layout_.normalize_packed_pressure(geometry, full);
        reflect_pressure(full);
        std::copy(full, full+size(), reduced);
    }

private:
    fdm::NSCylStateLayout<T> layout_;
    vector<long double> householder_;

    void reflect_pressure(T* state) const {
        long double dot = 0;
        for (int index = 0; index < layout_.p_size; ++index) {
            dot += householder_[index]*static_cast<long double>(
                state[layout_.p_offset+index]);
        }
        for (int index = 0; index < layout_.p_size; ++index) {
            state[layout_.p_offset+index] -= static_cast<T>(
                2*householder_[index]*dot);
        }
    }
};

template<typename T>
class GlobalCpuOperator {
public:
    using Task = fdm::NSCyl<T, false, fdm::tensor_flag::periodic>;

    explicit GlobalCpuOperator(const Config& config)
        : ns_(config)
        , gauge_(ns_.nr, ns_.nz, ns_.nphi, ns_.r0, ns_.dr)
        , operator_steps_(config.get("spectral", "operator_steps", 1))
        , full_(gauge_.full_size())
    {
        if (operator_steps_ <= 0) {
            throw std::invalid_argument("operator_steps must be positive");
        }
        gauge_.state_layout().initialize_couette_linearization(ns_);
        ns_.U0 = 0;
    }

    int size() const { return gauge_.size(); }
    const Task& geometry() const { return ns_; }

    void apply(T* y, const T* x) {
        gauge_.expand(ns_, x, full_.data());
        gauge_.state_layout().unpack(ns_, full_.data());
        for (int step = 0; step < operator_steps_; ++step) {
            ns_.L_step();
        }
        gauge_.state_layout().pack(ns_, full_.data());
        gauge_.reduce(ns_, full_.data(), y);
    }

private:
    Task ns_;
    GlobalGaugeLayout<T> gauge_;
    int operator_steps_;
    vector<T> full_;
};

#ifdef FDM_HAVE_SYCL
class GlobalSyclOperator {
public:
    GlobalSyclOperator(sycl::queue& queue, const Config& config)
        : queue_(queue)
        , ns_(queue,
              config.get("ns", "nr", 32),
              config.get("ns", "nz", 32),
              config.get("ns", "nphi", 32),
              config.get("ns", "r", static_cast<float>(M_PI/2)),
              config.get("ns", "R", static_cast<float>(M_PI)),
              config.get("ns", "h2", 10.0f)
                  -config.get("ns", "h1", 0.0f),
              0.0f,
              config.get("ns", "Re", 1.0f),
              config.get("ns", "dt", 0.001f))
        , gauge_(ns_.nr, ns_.nz, ns_.nphi, ns_.r0, ns_.dr)
        , operator_steps_(config.get("spectral", "operator_steps", 1))
        , full_(gauge_.full_size())
    {
        if (operator_steps_ <= 0) {
            throw std::invalid_argument("operator_steps must be positive");
        }
        ns_.initialize_couette_linearization(
            config.get("ns", "u0", 1.0f));
    }

    int size() const { return gauge_.size(); }
    const fdm::NSCylSycl<float>& geometry() const { return ns_; }

    void apply(float* y, const float* x) {
        queue_.wait();
        gauge_.expand(ns_, x, full_.data());
        unpack();
        for (int step = 0; step < operator_steps_; ++step) {
            ns_.L_step();
        }
        queue_.wait();
        pack();
        gauge_.reduce(ns_, full_.data(), y);
    }

private:
    sycl::queue& queue_;
    fdm::NSCylSycl<float> ns_;
    GlobalGaugeLayout<float> gauge_;
    int operator_steps_;
    vector<float> full_;

    void unpack() {
        auto u = ns_.ua();
        auto v = ns_.va();
        auto w = ns_.wa();
        auto p = ns_.pa();
        const auto& layout = gauge_.state_layout();
        int index = 0;
        for (int i = 0; i < ns_.nphi; ++i) {
            for (int k = 0; k < ns_.nz; ++k) {
                for (int j = 1; j < ns_.nr; ++j) {
                    u(i, k, j) = full_[index++];
                }
            }
        }
        if (index != layout.v_offset) {
            throw std::logic_error("invalid global SYCL u layout");
        }
        for (auto field : {v, w, p}) {
            for (int i = 0; i < ns_.nphi; ++i) {
                for (int k = 0; k < ns_.nz; ++k) {
                    for (int j = 1; j <= ns_.nr; ++j) {
                        field(i, k, j) = full_[index++];
                    }
                }
            }
        }
        if (index != layout.state_size) {
            throw std::logic_error("invalid global SYCL state layout");
        }
    }

    void pack() {
        auto u = ns_.ua();
        auto v = ns_.va();
        auto w = ns_.wa();
        auto p = ns_.pa();
        const auto& layout = gauge_.state_layout();
        int index = 0;
        for (int i = 0; i < ns_.nphi; ++i) {
            for (int k = 0; k < ns_.nz; ++k) {
                for (int j = 1; j < ns_.nr; ++j) {
                    full_[index++] = u(i, k, j);
                }
            }
        }
        if (index != layout.v_offset) {
            throw std::logic_error("invalid global SYCL u layout");
        }
        for (auto field : {v, w, p}) {
            for (int i = 0; i < ns_.nphi; ++i) {
                for (int k = 0; k < ns_.nz; ++k) {
                    for (int j = 1; j <= ns_.nr; ++j) {
                        full_[index++] = field(i, k, j);
                    }
                }
            }
        }
        if (index != layout.state_size) {
            throw std::logic_error("invalid global SYCL state layout");
        }
    }
};
#endif

template<typename T>
class GlobalFourierClassifier {
public:
    GlobalFourierClassifier(int nr, int nz, int nphi,
                            double r0, double dr)
        : gauge_(nr, nz, nphi, r0, dr)
        , fft_(nphi, nz)
        , values_(static_cast<std::size_t>(nphi)*nz)
        , coefficients_(values_.size())
        , full_(gauge_.full_size())
        , block_energy_(static_cast<std::size_t>(nphi/2+1)*(nz/2+1))
    { }

    template<typename Geometry>
    vector<double> energy(const Geometry& geometry, const T* reduced) {
        gauge_.expand(geometry, reduced, full_.data());
        std::fill(block_energy_.begin(), block_energy_.end(), 0.0);
        const auto& layout = gauge_.state_layout();
        add_component(layout.u_offset, layout.nr-1);
        add_component(layout.v_offset, layout.nr);
        add_component(layout.w_offset, layout.nr);
        return block_energy_;
    }

private:
    GlobalGaugeLayout<T> gauge_;
    fdm::PeriodicPackedFFT2<T> fft_;
    vector<T> values_;
    vector<T> coefficients_;
    vector<T> full_;
    vector<double> block_energy_;

    void add_component(int offset, int radial_count) {
        const auto& layout = gauge_.state_layout();
        for (int j = 0; j < radial_count; ++j) {
            for (int i = 0; i < layout.nphi; ++i) {
                for (int k = 0; k < layout.nz; ++k) {
                    values_[static_cast<std::size_t>(i)*layout.nz+k] =
                        full_[offset+(i*layout.nz+k)*radial_count+j];
                }
            }
            fft_.analysis(values_.data(), coefficients_.data());
            for (int i = 0; i < layout.nphi; ++i) {
                const int m = std::min(i, layout.nphi-i);
                for (int k = 0; k < layout.nz; ++k) {
                    const int l = std::min(k, layout.nz-k);
                    const double value = coefficients_[
                        static_cast<std::size_t>(i)*layout.nz+k];
                    block_energy_[static_cast<std::size_t>(m)*(layout.nz/2+1)+l]
                        += value*value;
                }
            }
        }
    }
};

template<typename T, typename Operator>
void run_global_impl(const Config& config, Operator& op) {
    const int n = op.size();
    const int nphi = config.get("ns", "nphi", 32);
    const int nz = config.get("ns", "nz", 32);
    const int operator_steps = config.get("spectral", "operator_steps", 1);
    const double dt = config.get("ns", "dt", 0.001);
    const double growth_tolerance =
        config.get("spectral", "growth_tol", 1e-8);
    const int stable_guard = std::max(1, config.get(
        "spectral", "global_stable_guard",
        config.get("spectral", "stable_guard", 4)));
    const int maxit = config.get("spectral", "maxit", 10000);
    const int residual_seed = config.get("spectral", "residual_seed", 0);
    const int starts = std::max(
        1, config.get("spectral", "global_starts", 1));
    const int minimum_passes = std::max(
        2, config.get("spectral", "global_min_passes", 2));
    const int confirmation_passes = std::max(
        1, config.get("spectral", "global_confirmation_passes", 1));
    const int requested_ncv = config.get("spectral", "global_ncv", 0);
    const int largest_valid_nev = n-2;
    int nev = std::min(
        config.get("spectral", "global_nev", 16), largest_valid_nev);
    const int max_nev = std::min(
        config.get("spectral", "global_max_nev", 96), largest_valid_nev);
    const int minimum_confirm_nev = std::min(std::max(
        1, config.get("spectral", "global_min_confirm_nev", 64)), max_nev);
    const double default_tolerance = std::max(
        1e-8, 8.0*static_cast<double>(std::numeric_limits<T>::epsilon()));
    const T tolerance = static_cast<T>(config.get(
        "spectral", "global_tol",
        config.get("spectral", "tol", default_tolerance)));
    const double default_energy_tolerance =
        std::is_same_v<T, float> ? 1e-2 : 1e-8;
    const double energy_tolerance = config.get(
        "spectral", "global_block_energy_tol", default_energy_tolerance);
    const int top_blocks = std::max(
        1, config.get("spectral", "global_top_blocks", 8));
    const bool require_reference_coverage = config.get(
        "spectral", "global_require_reference_coverage", 0) != 0;

    if (nev <= 0 || max_nev <= 0) {
        throw std::invalid_argument("global Arnoldi problem is too small");
    }
    if (!(dt > 0)) {
        throw std::invalid_argument("ns:dt must be positive");
    }
    if (!(energy_tolerance > 0 && energy_tolerance <= 1)) {
        throw std::invalid_argument(
            "global_block_energy_tol must be in (0,1]");
    }
    nev = std::min(nev, max_nev);
    const int nr = config.get("ns", "nr", 32);
    const double r0 = config.get("ns", "r", M_PI/2);
    const double radius = config.get("ns", "R", M_PI);
    GlobalFourierClassifier<T> classifier(
        nr, nz, nphi, r0, (radius-r0)/nr);
    std::map<std::pair<int, int>, double> candidates;
    std::set<std::pair<int, int>> reference_blocks;
    const string reference = config.get(
        "spectral", "global_reference", string());
    if (!reference.empty()) {
        fdm::NSCylSpectralModeSet<T> modes;
        fdm::NSCylSpectralMetadata metadata;
        const auto expected = fdm::make_ns_cyl_spectral_metadata<T>(config);
        fdm::NSCylSpectralStorage(reference).load(modes, metadata, expected);
        for (const auto& mode : modes.modes()) {
            reference_blocks.emplace(mode.m, mode.l);
        }
        printf("global reference: %s blocks=%zu real_dimension=%d\n",
               reference.c_str(), reference_blocks.size(),
               modes.real_dimension());
    }

    printf("NSCyl global physical-space ARPACK probe\n");
    printf("global_n=%d grid: nr=%d nz=%d nphi=%d Re=%.9g dt=%.9g "
           "operator_steps=%d\n",
           n, nr, nz, nphi,
           config.get("ns", "Re", 1.0), dt, operator_steps);
    printf("classification: velocity-only real-packed FFT "
           "block_energy_tol=%.3e\n", energy_tolerance);
    printf("adaptive Arnoldi: nev=%d max_nev=%d min_confirm_nev=%d "
           "stable_guard=%d starts=%d\n",
           nev, max_nev, minimum_confirm_nev, stable_guard, starts);

    int pass = 0;
    int unchanged_passes = 0;
    bool confirmed = false;
    for (;;) {
        ++pass;
        const std::size_t candidate_count_before = candidates.size();
        int ncv = requested_ncv > 0
            ? std::max(requested_ncv, nev+2)
            : std::max(2*nev+2, nev+8);
        ncv = std::min(ncv, n);
        if (ncv-nev < 2) {
            nev = ncv-2;
        }

        int best_stable_columns = 0;
        int best_nconv = 0;
        bool pass_converged = false;
        for (int start = 0; start < starts; ++start) {
            arpack_solver<T> solver(
                n, maxit, arpack_solver<T>::standard,
                arpack_solver<T>::largest_magnitude,
                arpack_solver<T>::fixed, tolerance);
            solver.set_ncv(ncv);
            vector<T> residual(n);
            const T seed = static_cast<T>(residual_seed+start);
            for (int i = 0; i < n; ++i) {
                const T x = static_cast<T>(i+1);
                residual[i] =
                    std::sin((T(0.371)+T(0.017)*seed)*x+T(0.131)*seed)
                    +T(0.5)*std::cos(
                        (T(0.193)+T(0.011)*seed)*x-T(0.073)*seed);
            }
            solver.set_resid(residual.data());

            vector<complex<T>> eigenvalues;
            vector<vector<T>> eigenvectors;
            int calls = 0;
            const auto before = std::chrono::steady_clock::now();
            solver.solve(
                [&](T* y, const T* x) {
                    op.apply(y, x);
                    ++calls;
                },
                eigenvalues, eigenvectors, nev);
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now()-before).count();
            best_nconv = std::max(best_nconv, solver.last_nconv());
            pass_converged = pass_converged
                || (solver.last_naupd_info() == 0
                    && solver.last_nconv() >= nev);

            printf("GLOBAL_START nev=%d ncv=%d start=%d calls=%d info=%d "
                   "iterations=%d nconv=%d seconds=%.3f\n",
                   nev, ncv, start, calls, solver.last_naupd_info(),
                   solver.last_iterations(), solver.last_nconv(), seconds);

            int stable_columns = 0;
            for (int i = 0; i < static_cast<int>(eigenvalues.size());) {
                const complex<T> value = eigenvalues[i];
                int columns = 1;
                if (value.imag() > tolerance
                    && i+1 < static_cast<int>(eigenvalues.size())
                    && eigenvalues[i+1].imag() < -tolerance) {
                    columns = 2;
                }
                const double magnitude = std::abs(value);
                const double growth = magnitude > 0
                    ? std::log(magnitude)/(operator_steps*dt)
                    : -INFINITY;
                printf("GLOBAL_RITZ start=%d index=%d columns=%d "
                       "abs=%.16e real=%.16e imag=%+.16e growth=%+.9e\n",
                       start, i, columns, magnitude,
                       static_cast<double>(value.real()),
                       static_cast<double>(value.imag()), growth);

                if (growth <= growth_tolerance) {
                    stable_columns += columns;
                    i += columns;
                    continue;
                }

                vector<double> energy = classifier.energy(
                    op.geometry(), eigenvectors[i].data());
                if (columns == 2) {
                    const vector<double> imaginary_energy = classifier.energy(
                        op.geometry(), eigenvectors[i+1].data());
                    for (std::size_t j = 0; j < energy.size(); ++j) {
                        energy[j] += imaginary_energy[j];
                    }
                }
                const double total = std::accumulate(
                    energy.begin(), energy.end(), 0.0);
                vector<int> order(energy.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    return energy[a] > energy[b];
                });
                const int printed = std::min(
                    top_blocks, static_cast<int>(order.size()));
                for (int rank = 0; rank < printed; ++rank) {
                    const int flat = order[rank];
                    const double share = total > 0 ? energy[flat]/total : 0;
                    const int m = flat/(nz/2+1);
                    const int l = flat%(nz/2+1);
                    printf("GLOBAL_BLOCK_SHARE start=%d ritz=%d rank=%d "
                           "m=%d l=%d share=%.9e\n",
                           start, i, rank, m, l, share);
                }
                for (int flat = 0; flat < static_cast<int>(energy.size());
                     ++flat) {
                    const double share = total > 0 ? energy[flat]/total : 0;
                    if (share >= energy_tolerance) {
                        const int m = flat/(nz/2+1);
                        const int l = flat%(nz/2+1);
                        auto& maximum = candidates[{m, l}];
                        maximum = std::max(maximum, share);
                    }
                }
                i += columns;
            }
            best_stable_columns = std::max(
                best_stable_columns, stable_columns);
        }

        const bool unchanged = candidates.size() == candidate_count_before;
        const bool guarded = best_stable_columns >= stable_guard;
        if (pass >= minimum_passes && nev >= minimum_confirm_nev
            && unchanged && guarded
            && pass_converged) {
            ++unchanged_passes;
        } else {
            unchanged_passes = 0;
        }
        std::size_t missing_reference = 0;
        for (const auto& block : reference_blocks) {
            missing_reference += candidates.count(block) == 0 ? 1 : 0;
        }
        const bool reference_complete = !reference_blocks.empty()
            && missing_reference == 0;
        confirmed = reference_complete
            || unchanged_passes >= confirmation_passes;

        printf("GLOBAL_PASS pass=%d nev=%d ncv=%d best_nconv=%d "
               "stable_columns=%d guard=%s converged=%s candidates=%zu "
               "new=%zu unchanged_passes=%d",
               pass, nev, ncv, best_nconv, best_stable_columns,
               best_stable_columns >= stable_guard ? "yes" : "no",
               pass_converged ? "yes" : "no", candidates.size(),
               candidates.size()-candidate_count_before, unchanged_passes);
        if (!reference_blocks.empty()) {
            printf(" reference_missing=%zu", missing_reference);
        }
        printf(" confirmed=%s\n", confirmed ? "yes" : "no");
        if (confirmed || nev >= max_nev) {
            break;
        }
        const int next_nev = std::min(max_nev, std::max(nev+2, 2*nev));
        if (next_nev == nev) {
            break;
        }
        nev = next_nev;
    }

    for (const auto& candidate : candidates) {
        printf("GLOBAL_CANDIDATE m=%d l=%d max_share=%.9e\n",
               candidate.first.first, candidate.first.second,
               candidate.second);
    }
    printf("global candidate blocks: %zu\n", candidates.size());
    if (!reference_blocks.empty()) {
        std::size_t missing = 0;
        for (const auto& block : reference_blocks) {
            if (candidates.count(block) == 0) {
                ++missing;
                printf("GLOBAL_REFERENCE_MISSING m=%d l=%d\n",
                       block.first, block.second);
            }
        }
        for (const auto& candidate : candidates) {
            if (reference_blocks.count(candidate.first) == 0) {
                printf("GLOBAL_REFERENCE_EXTRA m=%d l=%d\n",
                       candidate.first.first, candidate.first.second);
            }
        }
        if (require_reference_coverage && missing != 0) {
            throw std::runtime_error(
                "global Arnoldi missed blocks from the reference spectrum");
        }
    }
    if (!confirmed) {
        printf("warning: global candidate set reached global_max_nev "
               "without confirmation\n");
    }
    const string output = config.get("spectral", "output", string());
    if (!output.empty()) {
        printf("global probe does not write spectral output; ignored: %s\n",
               output.c_str());
    }
    printf("next step: verify GLOBAL_CANDIDATE blocks with the existing "
           "strategy=blocks dense path\n");
}

template<typename T>
void run_global(const Config& config) {
    const string backend = config.get(
        "spectral", "backend", string("cpu"));
    if (backend == "cpu") {
        GlobalCpuOperator<T> op(config);
        run_global_impl<T>(config, op);
        return;
    }
    if (backend != "sycl") {
        throw std::invalid_argument(
            "spectral backend must be either 'cpu' or 'sycl'");
    }
#ifdef FDM_HAVE_SYCL
    if constexpr (std::is_same_v<T, float>) {
        const sycl::device device = select_sycl_device();
        printf("SYCL device: %s\n",
               device.get_info<sycl::info::device::name>().c_str());
        sycl::queue queue{device, sycl::property::queue::in_order{}};
        GlobalSyclOperator op(queue, config);
        run_global_impl<float>(config, op);
    } else {
        throw std::invalid_argument(
            "SYCL spectral probe currently supports datatype=float only");
    }
#else
    throw std::runtime_error("spectral probe was built without SYCL");
#endif
}

template<typename T>
void run(const Config& config) {
    const string strategy = config.get(
        "spectral", "strategy", string("blocks"));
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
    const string backend = config.get(
        "spectral", "backend", string("cpu"));

    if (backend != "cpu" && backend != "sycl") {
        throw std::invalid_argument(
            "spectral backend must be either 'cpu' or 'sycl'");
    }
#ifndef FDM_HAVE_SYCL
    if (backend == "sycl") {
        throw std::runtime_error("spectral probe was built without SYCL");
    }
#endif

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
    threads = std::max(1, std::min(
        threads, static_cast<int>(blocks.size())));

    printf("NSCyl real-packed Fourier ARPACK probe\n");
    printf("grid: nr=%d nz=%d nphi=%d  Re=%.9g dt=%.9g "
           "r=[%.9g,%.9g] z=[%.9g,%.9g]\n",
           config.get("ns", "nr", 32), nz, nphi,
           config.get("ns", "Re", 1.0), dt,
           config.get("ns", "r", M_PI/2),
           config.get("ns", "R", M_PI),
           config.get("ns", "h1", 0.0),
           config.get("ns", "h2", 10.0));
    printf("blocks=%zu m=[%d,%d] l=[%d,%d] backend=%s strategy=%s "
           "threads=%d\n",
           blocks.size(), m_min, m_max, l_min, l_max,
           backend.c_str(), strategy.c_str(), threads);
    printf("selection: growth_tol=%.3e residual_tol=%.3e "
           "condition_limit=%.3e\n",
           growth_tolerance, residual_limit, condition_limit);
    printf("packing: q=cosine, N-q=sine; endpoints 0/Nyquist have one phase\n");

    vector<ProbeResult<T>> results(blocks.size());

    if (strategy == "batched_blocks") {
        probe_batched_blocks(config, blocks, results, backend);
    } else if (backend == "cpu") {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
#endif
        for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
            try {
                results[i] = probe_cpu_block<T>(config, blocks[i]);
            } catch (const std::exception& error) {
                results[i].block = blocks[i];
                results[i].error = error.what();
            }
        }
#ifdef FDM_HAVE_SYCL
    } else if constexpr (std::is_same_v<T, float>) {
        const sycl::device device = select_sycl_device();
        printf("SYCL device: %s\n", device
            .get_info<sycl::info::device::name>().c_str());
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
        {
            sycl::queue queue{
                device, sycl::property::queue::in_order{}};
#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
                try {
                    results[i] = probe_sycl_block(queue, config, blocks[i]);
                } catch (const std::exception& error) {
                    results[i].block = blocks[i];
                    results[i].error = error.what();
                }
            }
        }
#else
        sycl::queue queue{
            device, sycl::property::queue::in_order{}};
        for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
            try {
                results[i] = probe_sycl_block(queue, config, blocks[i]);
            } catch (const std::exception& error) {
                results[i].block = blocks[i];
                results[i].error = error.what();
            }
        }
#endif
    } else {
        throw std::invalid_argument(
            "SYCL spectral probe currently supports datatype=float only");
#endif
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
        } else if (result.guard_reached && !result.candidate) {
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
        const string strategy = config.get(
            "spectral", "strategy", string("blocks"));
        if (strategy != "blocks" && strategy != "global"
            && strategy != "batched_blocks") {
            throw std::invalid_argument(
                "spectral strategy must be 'blocks', 'batched_blocks', "
                "or 'global'");
        }
        if (datatype == "float") {
            if (strategy == "global") {
                run_global<float>(config);
            } else {
                run<float>(config);
            }
        } else {
            if (strategy == "global") {
                run_global<double>(config);
            } else {
                run<double>(config);
            }
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "spectral probe failed: %s\n", error.what());
        return 1;
    }
    return 0;
}
