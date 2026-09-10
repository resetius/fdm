// Nonlinear spectral filter: projection onto the stable manifold along the
// unstable subspace, compared with the linear projector.
//
// Method: dissertation, theoretical.tex "Окрестность стационарной точки" and
// practical.tex "Алгоритм со склейкой". The stable manifold is the graph
// W- = {w + f(w)} with
//
//     f(w) = P+ L^-1 ( f(P- S(w+f(w))) - P+ S(w+f(w)) ) + f(w),   f_0 = 0,
//
// evaluated by the memoized recursion, O(N^2) instead of O(2^N):
//
//     skleika(y, n, level):
//         if O[level,n] known: return it
//         if n <= 0: return 0
//         x1 = skleika(y, n-1, level)
//         y1 = P- S(x1+y);  x2 = P+ S(x1+y)
//         xx = skleika(y1, n-1, level+1)
//         x  = P+ L^-1 (xx - x2) + x1;  O[level,n] = x
//
// f_0 = 0 reproduces the linear filter exactly, so iteration 0 is the linear
// branch and later iterations measure what the nonlinear correction adds.
//
// S is the exact nonlinear perturbation map over map_steps steps,
// S(q) = step^k(Q_C+q) - Q_C, which keeps q = 0 because Q_C is stationary.
// The stored multipliers belong to the operator_steps-step operator, so they
// are raised to map_steps/operator_steps for L^-1.

#include <cmath>
#include <complex>
#include <cstdio>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// SYCL first: ns_cyl.h drags in declarations that clash with the AdaptiveCpp
// builtins header, the same order the sycl demo uses.
#ifdef FDM_HAVE_SYCL
#include <sycl/sycl.hpp>
#include "ns_cyl_sycl.h"
#endif

#include "config.h"
#include "ns_cyl.h"
#include "ns_cyl_nonlinear_gluing.h"
#include "ns_cyl_spectral_filter.h"
#include "ns_cyl_spectral_storage.h"
#include "ns_cyl_state.h"

namespace {

template<typename T>
using Task = fdm::NSCyl<T, false, fdm::tensor_flag::periodic>;

#ifdef FDM_HAVE_SYCL
// Presents NSCylSycl's USM fields the way NSCylStateLayout reads them --
// field[i][k][j] -- plus the flat (vec, size) pair clear_state fills.
template<typename T>
struct SyclField {
    fdm::CylAcc<T> acc;
    T* vec = nullptr;
    int size = 0;

    struct Row {
        fdm::CylAcc<T> acc;
        int i, k;
        T& operator[](int j) const { return acc(i, k, j); }
    };
    struct Plane {
        fdm::CylAcc<T> acc;
        int i;
        Row operator[](int k) const { return {acc, i, k}; }
    };
    Plane operator[](int i) const { return {acc, i}; }
};

template<typename T>
SyclField<T> sycl_field(fdm::CylAcc<T> acc, int nphi, int nz, int radial) {
    return {acc, acc.ptr, nphi*nz*radial};
}

sycl::queue& sycl_queue() {
    static sycl::queue queue{
        [] {
            for (const auto& platform : sycl::platform::get_platforms()) {
                for (const auto& device : platform.get_devices()) {
                    if (device.is_gpu()) { return device; }
                }
            }
            return sycl::device{sycl::cpu_selector_v};
        }(),
        sycl::property::queue::in_order{}};
    return queue;
}

// Same surface as Task<T> for what the layout and the branch loop touch,
// but step() runs on the device.
template<typename T>
class SyclTask {
public:
    explicit SyclTask(const Config& config)
        : nr(config.get("ns", "nr", 32))
        , nz(config.get("ns", "nz", 31))
        , nphi(config.get("ns", "nphi", 32))
        , dt(config.get("ns", "dt", 0.001))
        , ns_(sycl_queue(), nr, nz, nphi,
              T(config.get("ns", "r", M_PI/2)),
              T(config.get("ns", "R", M_PI)),
              T(config.get("ns", "h2", 10.0)-config.get("ns", "h1", 0.0)),
              T(config.get("ns", "u0", 1.0)),
              T(config.get("ns", "Re", 1.0)),
              T(dt))
        , u(sycl_field(ns_.ua(), nphi, nz, nr+3))
        , v(sycl_field(ns_.va(), nphi, nz, nr+2))
        , w(sycl_field(ns_.wa(), nphi, nz, nr+2))
        , p(sycl_field(ns_.pa(), nphi, nz, nr+2))
    { }

    void step() { ns_.step(); }
    void wait() { sycl_queue().wait(); }
    void apply_boundary_conditions() { ns_.apply_boundary_conditions(); }

    const int nr, nz, nphi;
    const double dt;

private:
    fdm::NSCylSycl<T> ns_;

public:
    SyclField<T> u, v, w, p;
};
#endif
template<typename T>
using Layout = fdm::NSCylStateLayout<T>;
template<typename T>
using Vector = std::vector<T>;

template<typename T>
Vector<T> add(const Vector<T>& a, const Vector<T>& b) {
    Vector<T> r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) { r[i] = a[i]+b[i]; }
    return r;
}

template<typename T>
Vector<T> sub(const Vector<T>& a, const Vector<T>& b) {
    Vector<T> r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) { r[i] = a[i]-b[i]; }
    return r;
}

// Device work is asynchronous, so drain before reading the fields back.
template<typename TaskT>
void drain(TaskT&) { }

#ifdef FDM_HAVE_SYCL
template<typename T>
void drain(SyclTask<T>& task) { task.wait(); }
#endif

template<typename T, typename StepTask = Task<T>>
class Method {
public:
    using value_type = T;

    Method(const Config& config, const Vector<T>& reference,
           fdm::NSCylSpectralFilter<T>& filter,
           std::vector<std::vector<std::complex<T>>> multipliers,
           int map_steps, double power)
        : config_(config), reference_(reference), filter_(filter)
        , multipliers_(std::move(multipliers))
        , map_steps_(map_steps), power_(power)
        , geometry_(config), layout_(geometry_)
    { }

    int size() const { return layout_.state_size; }

    Vector<T> zero() const { return Vector<T>(layout_.state_size, T(0)); }

    Vector<T> S(const Vector<T>& q) {
        StepTask run(config_);
        layout_.unpack_sum(run, reference_, q.data());
        for (int step = 0; step < map_steps_; ++step) { run.step(); }
        drain(run);
        ++applications_;
        return sub(layout_.pack(run), reference_);
    }

    Vector<T> Pminus(const Vector<T>& q) {
        Vector<T> state = add(reference_, q);
        filter_.remove_packed(geometry_, state, reference_);
        return sub(state, reference_);
    }

    Vector<T> Pplus(const Vector<T>& q) { return sub(q, Pminus(q)); }

    Vector<T> PplusLinv(const Vector<T>& q) {
        Vector<T> state = add(reference_, q);
        filter_.scale_unstable_packed(
            geometry_, state, reference_,
            [&](std::size_t block, std::vector<T>& c) {
                divide(multipliers_.at(block), c);
            });
        return sub(state, reference_);
    }

    fdm::NSCylSpectralFilterDiagnostics measure(const Vector<T>& q) {
        Vector<T> state = add(reference_, q);
        return filter_.measure_packed(geometry_, state, reference_);
    }

    long long applications() const { return applications_; }

private:
    // A complex pair (Re, Im) transforms under L as multiplication by
    // conj(mu), so the inverse divides by conj(mu^power). The power is a few
    // hundred, so mu^power is raised in double whatever the field type is.
    void divide(const std::vector<std::complex<T>>& mu, std::vector<T>& c) const {
        for (std::size_t i = 0; i < c.size(); ) {
            const std::complex<double> nu = std::pow(
                std::complex<double>(mu[i].real(), mu[i].imag()), power_);
            if (i+1 < c.size() && mu[i].imag() != T(0) && mu[i+1] == mu[i]) {
                const std::complex<double> z(c[i], c[i+1]);
                const std::complex<double> w = z/std::conj(nu);
                c[i] = static_cast<T>(w.real());
                c[i+1] = static_cast<T>(w.imag());
                i += 2;
            } else {
                c[i] = static_cast<T>(c[i]/nu.real()); i += 1;
            }
        }
    }

    const Config& config_;
    const Vector<T>& reference_;
    fdm::NSCylSpectralFilter<T>& filter_;
    std::vector<std::vector<std::complex<T>>> multipliers_;
    int map_steps_;
    double power_;
    Task<T> geometry_;
    Layout<T> layout_;
    long long applications_ = 0;
};

// Advance a perturbation by the exact nonlinear map for the given steps.
template<typename StepTask, typename T>
Vector<T> develop(const Config& config, const Vector<T>& reference,
                  const Vector<T>& q, int steps) {
    StepTask run(config);
    const Layout<T> layout(run);
    layout.unpack_sum(run, reference, q.data());
    for (int step = 0; step < steps; ++step) { run.step(); }
    drain(run);
    return sub(layout.pack(run), reference);
}

template<typename T>
Vector<T> couette_reference(const Config& config) {
    Task<T> state(config);
    const Layout<T> layout(state);
    layout.initialize_couette_state(state);
    return layout.pack(state);
}

template<typename T>
Vector<T> seed_perturbation(const Config& config, double epsilon) {
    Task<T> state(config);
    const Layout<T> layout(state);
    layout.initialize_couette_state(state);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<T> noise(-epsilon, epsilon);
    for (int i = 0; i < state.nphi; ++i) {
        for (int k = 0; k < state.nz; ++k) {
            for (int j = 1; j < state.nr; ++j) { state.u[i][k][j] += noise(rng); }
            for (int j = 1; j <= state.nr; ++j) {
                state.v[i][k][j] += noise(rng);
                state.w[i][k][j] += noise(rng);
            }
        }
    }
    return sub(layout.pack(state), couette_reference<T>(config));
}

// Propagate one branch and report when its unstable norm crosses the threshold.
template<typename StepTask, typename MethodT, typename T>
void run_branch(const Config& config, MethodT& method,
                const Vector<T>& reference,
                const Vector<T>& q0, const char* name, int steps, int interval,
                double unstable_threshold, FILE* csv) {
    StepTask run(config);
    const Layout<T> layout(run);
    layout.unpack_sum(run, reference, q0.data());
    double hold_time = -1;
    double last_unstable = 0;
    double last_time = 0;

    for (int step = 0; step <= steps; ++step) {
        if (step%interval == 0 || step == steps) {
            drain(run);
            const Vector<T> q = sub(layout.pack(run), reference);
            const auto d = method.measure(q);
            if (hold_time < 0 && d.removed_norm > unstable_threshold) {
                hold_time = step*run.dt;
            }
            last_unstable = d.removed_norm;
            last_time = step*run.dt;
            if (csv) {
                fprintf(csv, "%s,%d,%.9g,%.9e,%.9e\n", name, step,
                        step*run.dt, d.velocity_perturbation_norm,
                        d.removed_norm);
            }
        }
        if (step != steps) { run.step(); }
    }
    printf("  %-12s unstable(t=%.4g)=%.6e  crosses %.3g at t=%s\n",
           name, last_time, last_unstable, unstable_threshold,
           hold_time < 0 ? "not within window"
                         : std::to_string(hold_time).c_str());
}

template<typename T, typename StepTask = Task<T>>
int run(const Config& config) {
    const std::string spectrum = config.get(
        "nonlinear", "spectrum", std::string());
    if (spectrum.empty()) {
        fprintf(stderr, "nonlinear:spectrum=FILE.nc is required\n");
        return 1;
    }
    const int iterations = config.get("nonlinear", "iterations", 4);
    const int map_steps = config.get("nonlinear", "map_steps", 15000);
    const double epsilon = config.get("nonlinear", "epsilon", 1e-3);
    const int branch_steps = config.get("nonlinear", "branch_steps", 30000);
    const int interval = config.get("nonlinear", "interval", 1000);
    const double unstable_threshold = config.get(
        "nonlinear", "unstable_threshold", 1e-4);
    const std::string output = config.get(
        "nonlinear", "output", std::string());
    const int develop_steps = config.get("nonlinear", "develop_steps", 0);

    fdm::NSCylSpectralModeSet<T> modes;
    fdm::NSCylSpectralMetadata metadata;
    fdm::NSCylSpectralStorage(spectrum).load(modes, metadata);

    Task<T> probe(config);
    if (metadata.nr != probe.nr || metadata.nphi != probe.nphi
        || metadata.nz != probe.nz) {
        fprintf(stderr, "spectrum grid does not match the configuration\n");
        return 1;
    }
    const double power = static_cast<double>(map_steps)/metadata.operator_steps;

    fdm::NSCylSpectralFilter<T> filter(
        probe.nr, probe.nphi, probe.nz,
        fdm::NSCylSpectralProjector<T>(modes, metadata.condition_limit));

    const Vector<T> reference = couette_reference<T>(config);
    Method<T, StepTask> method(
        config, reference, filter, fdm::ns_cyl_block_multipliers(modes),
        map_steps, power);

    printf("nonlinear spectral filter\n");
    printf("grid %dx%dx%d  spectrum %s  blocks=%zu dimension=%d\n",
           probe.nr, probe.nphi, probe.nz, spectrum.c_str(),
           filter.projector().blocks().size(), modes.real_dimension());
    printf("map_steps=%d (T=%.4g)  stored operator_steps=%d  power=%.4g\n",
           map_steps, map_steps*probe.dt, metadata.operator_steps, power);
    printf("iterations=%d  epsilon=%.3g  branch_steps=%d\n\n",
           iterations, epsilon, branch_steps);
    fflush(stdout);

    FILE* csv = output.empty() ? nullptr : fopen(output.c_str(), "w");
    if (csv) { fprintf(csv, "branch,step,time,velocity,unstable\n"); }

    Vector<T> q0 = seed_perturbation<T>(config, epsilon);
    if (develop_steps > 0) {
        q0 = develop<StepTask>(config, reference, q0, develop_steps);
        const auto d = method.measure(q0);
        printf("checkpoint at t=%.4g: velocity=%.9e unstable=%.9e\n\n",
               develop_steps*probe.dt, d.velocity_perturbation_norm,
               d.removed_norm);
        fflush(stdout);
    }
    const Vector<T> y = method.Pminus(q0);

    run_branch<StepTask>(config, method, reference, q0, "unfiltered",
                         branch_steps, interval, unstable_threshold, csv);

    fdm::NSCylNonlinearGluing<Method<T, StepTask>> glue(method);
    for (int n = 0; n <= iterations; ++n) {
        const Vector<T> x = glue(y, n, 0);
        const Vector<T> projected = add(y, x);
        const auto d = method.measure(projected);
        const std::string name = n == 0 ? "linear" : ("nonlinear"+std::to_string(n));
        printf("  f^%d: ||x||=%.6e  unstable=%.6e  S applications=%lld\n",
               n, method.measure(x).velocity_perturbation_norm,
               d.removed_norm, method.applications());
        fflush(stdout);
        run_branch<StepTask>(config, method, reference, projected, name.c_str(),
                             branch_steps, interval, unstable_threshold, csv);
    }

    if (csv) { fclose(csv); }
    printf("\ntotal S applications: %lld\n", method.applications());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string file;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-c" && i+1 < argc) { file = argv[++i]; }
    }
    Config config;
    if (!file.empty()) { config.open(file); }
    config.rewrite(argc, argv);

    const std::string datatype = config.get(
        "solver", "datatype", std::string("double"));
    const std::string backend = config.get(
        "nonlinear", "backend", std::string("cpu"));
    printf("datatype=%s backend=%s\n", datatype.c_str(), backend.c_str());

    if (backend == "sycl") {
#ifdef FDM_HAVE_SYCL
        if (datatype != "float") {
            fprintf(stderr, "the sycl backend supports datatype=float only\n");
            return 1;
        }
        printf("SYCL device: %s\n",
               sycl_queue().get_device()
                   .get_info<sycl::info::device::name>().c_str());
        return run<float, SyclTask<float>>(config);
#else
        fprintf(stderr, "this build has no SYCL\n");
        return 1;
#endif
    }
    if (backend != "cpu") {
        fprintf(stderr, "nonlinear:backend must be 'cpu' or 'sycl'\n");
        return 1;
    }
    if (datatype == "float") { return run<float>(config); }
    if (datatype == "double") { return run<double>(config); }
    fprintf(stderr, "solver:datatype must be 'float' or 'double'\n");
    return 1;
}
