#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "config.h"
#include "ns_cyl.h"
#include "ns_cyl_fourier_batch_sycl.h"
#include "ns_cyl_fourier_block_sycl.h"
#include "ns_cyl_spectral_modes.h"
#include "ns_cyl_sycl.h"

extern "C" {
#include <cmocka.h>
}

using fdm::NSCylSycl;

namespace {

constexpr int kNr = 8, kNz = 8, kNphi = 8;
constexpr float kR0 = 1.0f, kR = 2.0f, kLz = float(2*M_PI);
constexpr float kU0 = 1.0f, kRe = 10.0f, kDt = 1e-3f;

// Same device choice as the demo: the real deployment path is the GPU, and
// Metal has no fp64, so the kernels are exercised in float.
sycl::queue& queue() {
    static sycl::queue q{
        []() {
            for (auto& platform : sycl::platform::get_platforms())
                for (auto& device : platform.get_devices())
                    if (device.is_gpu()) return device;
            return sycl::device{sycl::cpu_selector_v};
        }(),
        sycl::property::queue::in_order{}};
    return q;
}

// A z-dependent, azimuthally varying state with zero radial velocity on both
// cylinder walls -- the same shape used by the CPU test in ut_ns_cyl.cpp.
void fill_smooth_state(NSCylSycl<float>& ns) {
    auto u = ns.ua(), v = ns.va(), w = ns.wa(), p = ns.pa();
    const double couette_a = -double(kU0)*kR0/(double(kR)*kR-double(kR0)*kR0);
    const double couette_b =
        double(kU0)*kR0*kR*kR/(double(kR)*kR-double(kR0)*kR0);

    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = 0; k < ns.nz; ++k) {
            for (int j = 1; j <= ns.nr; ++j) {
                const double r = ns.r0+(j-0.5)*ns.dr;
                w(i,k,j) = float(couette_a*r+couette_b/r
                    +0.05*std::cos(2*M_PI*i/ns.nphi)*std::sin(0.7*k));
                v(i,k,j) = float(0.02*std::cos(4*M_PI*i/ns.nphi)
                    *std::sin(M_PI*(j-0.5)/ns.nr)*std::sin(0.3*k));
                p(i,k,j) = float(0.02*std::sin(2*M_PI*i/ns.nphi+0.4*k));
            }
            for (int j = 1; j < ns.nr; ++j) {
                u(i,k,j) = float(0.03*std::sin(2*M_PI*i/ns.nphi)
                    *std::sin(M_PI*j/ns.nr)*std::cos(0.5*k));
            }
        }
    }
}

Config make_cpu_config(double reynolds = kRe) {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl_sycl",
        "--ns:r=1.0",
        "--ns:R=2.0",
        "--ns:h1=0.0",
        "--ns:h2=6.2831854820251465",
        "--ns:nr=8",
        "--ns:nz=8",
        "--ns:nphi=8",
        "--ns:u0=1.0",
        "--ns:Re="+std::to_string(reynolds),
        "--ns:dt=0.0010000000474974513",
        "--ns:verbose=0"
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    config.rewrite(static_cast<int>(argv.size()), argv.data());
    return config;
}

Config make_re100_n16_config() {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl_sycl",
        "--ns:r=1.5707963267948966",
        "--ns:R=3.141592653589793",
        "--ns:h1=0.0",
        "--ns:h2=10.0",
        "--ns:nr=16",
        "--ns:nz=16",
        "--ns:nphi=16",
        "--ns:u0=1.0",
        "--ns:Re=100.0",
        "--ns:dt=0.001",
        "--ns:verbose=0"
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    config.rewrite(static_cast<int>(argv.size()), argv.data());
    return config;
}

void copy_state_to_cpu(
    const NSCylSycl<float>& source,
    fdm::NSCyl<float, false, fdm::tensor_flag::periodic>& destination)
{
    auto u = source.ua();
    auto v = source.va();
    auto w = source.wa();
    auto p = source.pa();
    for (int i = 0; i < source.nphi; ++i) {
        for (int k = 0; k < source.nz; ++k) {
            for (int j = -1; j <= source.nr+1; ++j) {
                destination.u[i][k][j] = u(i, k, j);
            }
            for (int j = 0; j <= source.nr+1; ++j) {
                destination.v[i][k][j] = v(i, k, j);
                destination.w[i][k][j] = w(i, k, j);
                destination.p[i][k][j] = p(i, k, j);
            }
        }
    }
}

struct Difference {
    double error = 0;
    double scale = 0;

    void add(double actual, double expected) {
        error = std::max(error, std::abs(actual-expected));
        scale = std::max(scale, std::abs(expected));
    }

    double relative() const {
        return scale > 0 ? error/scale : error;
    }
};

void check_sycl_poisson_matches_cpu(
    int nr, int nz, int nphi, float r0, float outer_r, float lz)
{
    const float dr = (outer_r-r0)/nr;
    const float dz = lz/nz;
    const float pressure_r0 = r0-dr/2;
    const int size = nphi*nz*nr;
    fdm::LaplCyl3FFT2<
        float, false, fdm::tensor_flag::periodic> cpu(
            dr, dz, pressure_r0, outer_r-r0+dr, lz,
            nr, nz, nphi);
    fdm::LaplCylSycl<float> device(
        queue(), nr, nz, nphi, pressure_r0, dr, dz, lz);

    std::vector<float> rhs(size);
    std::vector<float> cpu_answer(size);
    float* device_rhs = sycl::malloc_shared<float>(size, queue());
    float* device_answer = sycl::malloc_shared<float>(size, queue());
    for (int i = 0; i < nphi; ++i) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < nr; ++j) {
                const int index = (i*nz+k)*nr+j;
                rhs[index] = float(
                    std::sin(0.37*(i+1)+0.19*(j+1))
                    +0.4*std::cos(0.53*(k+1)-0.11*(j+1)));
                device_rhs[index] = rhs[index];
            }
        }
    }

    cpu.solve(cpu_answer.data(), rhs.data());
    device.solve(device_answer, device_rhs);
    queue().wait();

    Difference difference;
    for (int index = 0; index < size; ++index) {
        difference.add(device_answer[index], cpu_answer[index]);
    }
    printf("SYCL/CPU float Poisson %dx%dx%d relative error: %e\n",
           nr, nz, nphi, difference.relative());
    sycl::free(device_answer, queue());
    sycl::free(device_rhs, queue());
    assert_true(difference.relative() < 2e-5);
}

void test_sycl_poisson_matches_cpu_float_reference(void**) {
    check_sycl_poisson_matches_cpu(8, 8, 8, 1.0f, 2.0f, kLz);
    check_sycl_poisson_matches_cpu(9, 8, 8, 1.0f, 2.0f, kLz);
    check_sycl_poisson_matches_cpu(16, 16, 16, kR0, kR, 10.0f);
    check_sycl_poisson_matches_cpu(32, 32, 32, kR0, kR, 10.0f);
}

void check_sycl_fourier_block_poisson_matches_full(int n, int m, int l) {
    const float r0 = 1.0f;
    const float outer_r = 2.0f;
    const float lz = 10.0f;
    const float dr = (outer_r-r0)/n;
    const int size = n*n*n;
    fdm::LaplCylSycl<float> solver(
        queue(), n, n, n, r0-dr/2, dr, lz/n, lz);
    float* rhs = sycl::malloc_shared<float>(size, queue());
    float* full = sycl::malloc_shared<float>(size, queue());
    float* block = sycl::malloc_shared<float>(size, queue());

    for (int i = 0; i < n; ++i) {
        const double phi = 2*M_PI*m*i/n;
        for (int k = 0; k < n; ++k) {
            const double z = 2*M_PI*l*k/n;
            const double phase =
                0.7*std::cos(phi)*std::cos(z)
                +0.3*std::cos(phi)*std::sin(z)
                -0.2*std::sin(phi)*std::cos(z)
                +0.4*std::sin(phi)*std::sin(z);
            for (int j = 0; j < n; ++j) {
                const double radial = std::sin(0.27*(j+1))
                    +0.2*std::cos(0.11*(j+1));
                rhs[(i*n+k)*n+j] = static_cast<float>(phase*radial);
            }
        }
    }

    solver.solve(full, rhs);
    queue().wait();
    solver.solve_fourier_block(block, rhs, m, l);
    queue().wait();

    Difference difference;
    for (int index = 0; index < size; ++index) {
        difference.add(block[index], full[index]);
    }
    printf("SYCL block/full Poisson n=%d (%d,%d) relative error: %e\n",
           n, m, l, difference.relative());
    sycl::free(block, queue());
    sycl::free(full, queue());
    sycl::free(rhs, queue());
    assert_true(difference.relative() < 2e-5);
}

void test_sycl_fourier_block_poisson_matches_full(void**) {
    check_sycl_fourier_block_poisson_matches_full(8, 0, 0);
    check_sycl_fourier_block_poisson_matches_full(8, 0, 3);
    check_sycl_fourier_block_poisson_matches_full(8, 2, 3);
    check_sycl_fourier_block_poisson_matches_full(8, 4, 4);
    check_sycl_fourier_block_poisson_matches_full(32, 2, 3);
}

void test_sycl_step_matches_cpu_float_reference(void**) {
    using CpuNS = fdm::NSCyl<float, false, fdm::tensor_flag::periodic>;
    CpuNS cpu(make_cpu_config());
    NSCylSycl<float> sycl(
        queue(), kNr, kNz, kNphi, kR0, kR, kLz, kU0, kRe, kDt);
    fill_smooth_state(sycl);
    copy_state_to_cpu(sycl, cpu);

    cpu.step();
    sycl.step();
    queue().wait();

    auto u = sycl.ua();
    auto v = sycl.va();
    auto w = sycl.wa();
    auto p = sycl.pa();
    auto x = sycl.xa();
    auto F = sycl.Fa();
    auto G = sycl.Ga();
    auto H = sycl.Ha();
    auto rhs = sycl.Ra();
    Difference dF, dG, dH, drhs, drhs_formula, dx, dp_boundary;
    Difference du, dv, dw, dp;
    for (int i = 0; i < kNphi; ++i) {
        for (int k = 0; k < kNz; ++k) {
            dp_boundary.add(p(i, k, 0), cpu.p[i][k][0]);
            dp_boundary.add(p(i, k, kNr+1), cpu.p[i][k][kNr+1]);
            for (int j = 0; j <= kNr; ++j) {
                dF.add(F(i, k, j), cpu.F[i][k][j]);
            }
            for (int j = 1; j < kNr; ++j) {
                du.add(u(i, k, j), cpu.u[i][k][j]);
            }
            for (int j = 1; j <= kNr; ++j) {
                dG.add(G(i, k, j), cpu.G[i][k][j]);
                dH.add(H(i, k, j), cpu.H[i][k][j]);
                drhs.add(rhs(i, k, j), cpu.RHS[i][k][j]);
                const float r = sycl.r0+sycl.dr*j-sycl.dr*0.5f;
                float reconstructed_rhs =
                    (((r+sycl.dr*0.5f)*F(i,k,j)
                       -(r-sycl.dr*0.5f)*F(i,k,j-1))/(r*sycl.dr)
                     +(G(i,k,j)-G(i,k-1,j))/sycl.dz
                     +(H(i,k,j)-H(i-1,k,j))/(r*sycl.dphi))/sycl.dt;
                if (j == 1) {
                    reconstructed_rhs -= (r-sycl.dr*0.5f)/r
                        *p(i,k,0)/sycl.dr2;
                }
                if (j == kNr) {
                    reconstructed_rhs -= (r+sycl.dr*0.5f)/r
                        *p(i,k,kNr+1)/sycl.dr2;
                }
                drhs_formula.add(rhs(i, k, j), reconstructed_rhs);
                dx.add(x(i, k, j), cpu.x[i][k][j]);
                dv.add(v(i, k, j), cpu.v[i][k][j]);
                dw.add(w(i, k, j), cpu.w[i][k][j]);
                dp.add(p(i, k, j), cpu.p[i][k][j]);
            }
        }
    }

    printf("SYCL/CPU float error (absolute / relative): "
           "F=%e/%e G=%e/%e H=%e/%e\n",
           dF.error, dF.relative(), dG.error, dG.relative(),
           dH.error, dH.relative());
    printf("SYCL/CPU float error (absolute / relative): "
           "p_bound=%e/%e RHS=%e/%e x=%e/%e\n",
           dp_boundary.error, dp_boundary.relative(),
           drhs.error, drhs.relative(), dx.error, dx.relative());
    printf("SYCL RHS kernel/host-float error (absolute / relative): "
           "%e/%e\n", drhs_formula.error, drhs_formula.relative());
    printf("SYCL/CPU float error (absolute / relative): "
           "u=%e/%e v=%e/%e w=%e/%e p=%e/%e\n",
           du.error, du.relative(), dv.error, dv.relative(),
           dw.error, dw.relative(), dp.error, dp.relative());
    assert_true(dF.error < 2e-6);
    assert_true(dG.error < 1e-6);
    assert_true(dH.error < 2e-6);
    assert_true(dp_boundary.error < 5e-8);
    assert_true(drhs_formula.relative() < 1e-6);
    assert_true(drhs.relative() < 2e-2);
    assert_true(dx.relative() < 2e-2);
    assert_true(du.error < 4e-4);
    assert_true(dv.error < 4e-4);
    assert_true(dw.error < 1e-4);
    assert_true(dp.relative() < 2e-2);
}

std::vector<float> make_block_input(int size) {
    std::vector<float> input(size);
    for (int index = 0; index < size; ++index) {
        const float x = static_cast<float>(index+1);
        input[index] = 0.07f*std::sin(0.173f*x)
            +0.03f*std::cos(0.317f*x+0.11f);
    }
    return input;
}

void check_sycl_linear_block_matches_cpu(int m, int l) {
    Config config = make_cpu_config();
    fdm::NSCylFourierBlockReference<float, true> cpu(config, m, l);
    fdm::NSCylSyclFourierBlockReference<float> device(
        queue(), config, m, l);
    assert_int_equal(device.size(), cpu.size());

    std::vector<float> input = make_block_input(cpu.size());
    std::vector<float> cpu_image(cpu.size());
    std::vector<float> device_image(cpu.size());

    device.lift(input.data());
    auto initial_u = device.task().ua();
    auto initial_v = device.task().va();
    auto initial_w = device.task().wa();
    std::vector<float> u_before(kNphi*kNz*(kNr+3));
    std::vector<float> v_before(kNphi*kNz*(kNr+2));
    std::vector<float> w_before(kNphi*kNz*(kNr+2));
    auto u_index = [](int i, int k, int j) {
        return (i*kNz+k)*(kNr+3)+j+1;
    };
    auto centered_index = [](int i, int k, int j) {
        return (i*kNz+k)*(kNr+2)+j;
    };
    for (int i = 0; i < kNphi; ++i) {
        for (int k = 0; k < kNz; ++k) {
            for (int j = -1; j <= kNr+1; ++j) {
                u_before[u_index(i,k,j)] = initial_u(i,k,j);
            }
            for (int j = 0; j <= kNr+1; ++j) {
                v_before[centered_index(i,k,j)] = initial_v(i,k,j);
                w_before[centered_index(i,k,j)] = initial_w(i,k,j);
            }
        }
    }

    cpu.apply(cpu_image.data(), input.data());
    device.apply(device_image.data(), input.data());

    auto& cpu_task = cpu.task();
    auto& device_task = device.task();
    auto device_w0 = device_task.w0a();
    auto device_F = device_task.Fa();
    auto device_G = device_task.Ga();
    auto device_H = device_task.Ha();
    auto device_rhs = device_task.Ra();
    auto device_x = device_task.xa();
    auto device_p = device_task.pa();
    Difference base, dF, dG, dH, drhs, drhs_formula, dx;
    for (int i = 0; i < kNphi; ++i) {
        for (int k = 0; k < kNz; ++k) {
            for (int j = 0; j <= kNr+1; ++j) {
                base.add(device_w0(i,k,j), cpu_task.w0[i][k][j]);
            }
            for (int j = 0; j <= kNr; ++j) {
                dF.add(
                    u_before[u_index(i,k,j)]+kDt*device_F(i,k,j),
                    cpu_task.F[i][k][j]);
            }
            for (int j = 1; j <= kNr; ++j) {
                dG.add(
                    v_before[centered_index(i,k,j)]+kDt*device_G(i,k,j),
                    cpu_task.G[i][k][j]);
                dH.add(
                    w_before[centered_index(i,k,j)]+kDt*device_H(i,k,j),
                    cpu_task.H[i][k][j]);
                drhs.add(device_rhs(i,k,j)/kDt, cpu_task.RHS[i][k][j]);
                const float r = device_task.r0+device_task.dr*j
                    -device_task.dr*0.5f;
                const int im = (i+kNphi-1)%kNphi;
                const int km = (k+kNz-1)%kNz;
                const float velocity_divergence =
                    ((r+device_task.dr*0.5f)
                         *u_before[u_index(i,k,j)]
                     -(r-device_task.dr*0.5f)
                         *u_before[u_index(i,k,j-1)])
                         /(r*device_task.dr)
                    +(v_before[centered_index(i,k,j)]
                      -v_before[centered_index(i,km,j)])/device_task.dz
                    +(w_before[centered_index(i,k,j)]
                      -w_before[centered_index(im,k,j)])
                        /(r*device_task.dphi);
                const float tendency_divergence =
                    ((r+device_task.dr*0.5f)*device_F(i,k,j)
                     -(r-device_task.dr*0.5f)*device_F(i,k,j-1))
                        /(r*device_task.dr)
                    +(device_G(i,k,j)-device_G(i,k-1,j))/device_task.dz
                    +(device_H(i,k,j)-device_H(i-1,k,j))
                        /(r*device_task.dphi);
                float reconstructed_rhs = velocity_divergence
                    +device_task.dt*tendency_divergence;
                if (j == 1) {
                    reconstructed_rhs -= (r-device_task.dr*0.5f)/r
                        *device_p(i,k,0)/device_task.dr2;
                }
                if (j == kNr) {
                    reconstructed_rhs -= (r+device_task.dr*0.5f)/r
                        *device_p(i,k,kNr+1)/device_task.dr2;
                }
                drhs_formula.add(device_rhs(i,k,j), reconstructed_rhs);
                dx.add(device_x(i,k,j)/kDt, cpu_task.x[i][k][j]);
            }
        }
    }

    Difference velocity, pressure;
    const int radial_size = cpu.radial_size();
    const int pressure_offset = 3*kNr-1;
    for (int index = 0; index < cpu.size(); ++index) {
        const int radial_index = index%radial_size;
        if (radial_index < pressure_offset) {
            velocity.add(device_image[index], cpu_image[index]);
        } else {
            pressure.add(device_image[index], cpu_image[index]);
        }
    }
    printf("SYCL/CPU L_step block (%d,%d): velocity=%e/%e "
           "pressure=%e/%e leakage=%e/%e\n",
           m, l, velocity.error, velocity.relative(),
           pressure.error, pressure.relative(),
           device.last_fourier_leakage(), cpu.last_fourier_leakage());
    printf("SYCL/CPU L_step stages (%d,%d): base=%e/%e "
           "F=%e/%e G=%e/%e H=%e/%e RHS=%e/%e "
           "RHS-formula=%e/%e x=%e/%e\n",
           m, l, base.error, base.relative(),
           dF.error, dF.relative(), dG.error, dG.relative(),
           dH.error, dH.relative(), drhs.error, drhs.relative(),
           drhs_formula.error, drhs_formula.relative(),
           dx.error, dx.relative());
    assert_true(velocity.relative() < 5e-2);
    assert_true(pressure.relative() < 2e-2);
    assert_true(dF.error < 1e-5);
    assert_true(dG.error < 1e-5);
    assert_true(dH.error < 1e-5);
    assert_true(drhs_formula.relative() < 1e-6);
    assert_true(device.last_fourier_leakage() < 5e-5);
}

void test_sycl_linear_fourier_blocks_match_cpu(void**) {
    check_sycl_linear_block_matches_cpu(0, 0);
    check_sycl_linear_block_matches_cpu(0, 3);
    check_sycl_linear_block_matches_cpu(1, 3);
    check_sycl_linear_block_matches_cpu(4, 4);
}

void test_sycl_batched_blocks_match_individual_applications(void**) {
    Config config = make_cpu_config();
    constexpr int operator_steps = 3;
    const std::vector<std::pair<int, int>> indices = {
        {0, 0}, {0, 3}, {1, 3}, {4, 4}
    };
    const float scales[] = {1e-3f, 1.0f, 1e2f, 1e3f};
    fdm::NSCylSyclFourierBlockBatchReference<float> batch(
        queue(), config, operator_steps);
    std::vector<std::vector<float>> inputs(indices.size());
    std::vector<std::vector<float>> batched(indices.size());
    std::vector<std::vector<float>> individual(indices.size());
    std::vector<fdm::NSCylFourierBatchRequest<float>> requests;

    for (std::size_t block_index = 0;
         block_index < indices.size(); ++block_index) {
        const auto [m, l] = indices[block_index];
        fdm::NSCylSyclFourierBlockReference<float> block(
            queue(), config, m, l, operator_steps);
        inputs[block_index] = make_block_input(block.size());
        for (float& value : inputs[block_index]) {
            value *= scales[block_index];
        }
        batched[block_index].resize(block.size());
        individual[block_index].resize(block.size());
        block.apply(
            individual[block_index].data(), inputs[block_index].data());
        requests.push_back({
            m, l, inputs[block_index].data(), batched[block_index].data(),
            block.size()});
    }

    batch.apply(requests);
    for (std::size_t block_index = 0;
         block_index < indices.size(); ++block_index) {
        Difference difference;
        for (std::size_t i = 0; i < batched[block_index].size(); ++i) {
            difference.add(
                batched[block_index][i], individual[block_index][i]);
        }
        printf("SYCL batched/individual block (%d,%d): %e/%e\n",
               indices[block_index].first, indices[block_index].second,
               difference.error, difference.relative());
        assert_true(difference.relative() < 2e-4);
    }
}

void initialize_nonlinear_couette(NSCylSycl<float>& ns) {
    const auto velocity = fdm::make_discrete_couette_velocity<float>(ns);
    const auto pressure = fdm::make_discrete_couette_pressure(ns, velocity);
    auto w = ns.wa();
    auto p = ns.pa();
    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = 0; k < ns.nz; ++k) {
            for (int j = 0; j <= ns.nr+1; ++j) {
                w(i,k,j) = velocity[j];
                p(i,k,j) = pressure[j];
            }
        }
    }
}

void test_sycl_linear_step_matches_centered_nonlinear_difference(void**) {
    Config config = make_cpu_config();
    fdm::NSCylSyclFourierBlockReference<float> linear(
        queue(), config, 0, 3);
    std::vector<float> input = make_block_input(linear.size());
    std::vector<float> linear_image(linear.size());
    std::vector<float> centered_image(linear.size());
    linear.apply(linear_image.data(), input.data());

    linear.lift(input.data());
    auto& perturbation = linear.task();
    NSCylSycl<float> plus(
        queue(), kNr, kNz, kNphi, kR0, kR, kLz, kU0, kRe, kDt);
    NSCylSycl<float> minus(
        queue(), kNr, kNz, kNphi, kR0, kR, kLz, kU0, kRe, kDt);
    initialize_nonlinear_couette(plus);
    initialize_nonlinear_couette(minus);

    constexpr float epsilon = 0.125f;
    auto qu = perturbation.ua();
    auto qv = perturbation.va();
    auto qw = perturbation.wa();
    auto qp = perturbation.pa();
    auto pu = plus.ua();
    auto pv = plus.va();
    auto pw = plus.wa();
    auto pp = plus.pa();
    auto mu = minus.ua();
    auto mv = minus.va();
    auto mw = minus.wa();
    auto mp = minus.pa();
    for (int i = 0; i < kNphi; ++i) {
        for (int k = 0; k < kNz; ++k) {
            for (int j = 1; j < kNr; ++j) {
                pu(i,k,j) += epsilon*qu(i,k,j);
                mu(i,k,j) -= epsilon*qu(i,k,j);
            }
            for (int j = 1; j <= kNr; ++j) {
                pv(i,k,j) += epsilon*qv(i,k,j);
                pw(i,k,j) += epsilon*qw(i,k,j);
                pp(i,k,j) += epsilon*qp(i,k,j);
                mv(i,k,j) -= epsilon*qv(i,k,j);
                mw(i,k,j) -= epsilon*qw(i,k,j);
                mp(i,k,j) -= epsilon*qp(i,k,j);
            }
        }
    }

    plus.step();
    minus.step();
    queue().wait();

    const float scale = 1/(2*epsilon);
    for (int i = 0; i < kNphi; ++i) {
        for (int k = 0; k < kNz; ++k) {
            for (int j = 1; j < kNr; ++j) {
                qu(i,k,j) = scale*(pu(i,k,j)-mu(i,k,j));
            }
            for (int j = 1; j <= kNr; ++j) {
                qv(i,k,j) = scale*(pv(i,k,j)-mv(i,k,j));
                qw(i,k,j) = scale*(pw(i,k,j)-mw(i,k,j));
                qp(i,k,j) = scale*(pp(i,k,j)-mp(i,k,j));
            }
        }
    }
    linear.extract(centered_image.data());

    Difference difference;
    for (int index = 0; index < linear.size(); ++index) {
        difference.add(centered_image[index], linear_image[index]);
    }
    printf("SYCL L_step/centered nonlinear derivative block (0,3): "
           "%e/%e\n", difference.error, difference.relative());
    assert_true(difference.relative() < 2e-3);
}

template<typename From, typename To>
double directed_eigenvalue_distance_near_unit_circle(
    const std::vector<std::complex<From>>& from,
    const std::vector<std::complex<To>>& to,
    double minimum_magnitude)
{
    double result = 0;
    for (const auto& value : from) {
        if (std::abs(value) < minimum_magnitude) {
            continue;
        }
        double nearest = std::numeric_limits<double>::infinity();
        for (const auto& candidate : to) {
            const std::complex<double> delta{
                static_cast<double>(value.real())
                    -static_cast<double>(candidate.real()),
                static_cast<double>(value.imag())
                    -static_cast<double>(candidate.imag())};
            nearest = std::min(nearest, std::abs(delta));
        }
        result = std::max(result, nearest);
    }
    return result;
}

template<typename T>
void print_largest_eigenvalues(
    const char* label, const std::vector<std::complex<T>>& values)
{
    std::vector<std::complex<T>> sorted = values;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return std::abs(a) > std::abs(b);
    });
    printf("%s largest:", label);
    for (int i = 0; i < std::min<int>(8, sorted.size()); ++i) {
        printf(" (%+.8e,%+.8e)|%e", static_cast<double>(sorted[i].real()),
               static_cast<double>(sorted[i].imag()),
               static_cast<double>(std::abs(sorted[i])));
    }
    printf("\n");
}

void test_sycl_linear_block_dense_spectrum_matches_cpu(void**) {
    Config config = make_cpu_config();
    constexpr int operator_steps = 4;
    fdm::NSCylFourierBlockReference<float, true> cpu(
        config, 0, 3, operator_steps);
    fdm::NSCylFourierBlockReference<double, true> reference(
        config, 0, 3, operator_steps);
    fdm::NSCylSyclFourierBlockReference<float> device(
        queue(), config, 0, 3, operator_steps);

    const auto cpu_spectrum = fdm::solve_ns_cyl_dense_block(
        cpu, kDt, 0.0, 1e-4);
    const auto device_spectrum = fdm::solve_ns_cyl_dense_block(
        device, kDt, 0.0, 1e-4);
    const auto reference_spectrum = fdm::solve_ns_cyl_dense_block(
        reference, kDt, 0.0, 1e-10);
    constexpr double minimum_magnitude = 0.9;
    const auto symmetric_near_unit_distance = [&](const auto& a,
                                                   const auto& b) {
        return std::max(
            directed_eigenvalue_distance_near_unit_circle(
                a, b, minimum_magnitude),
            directed_eigenvalue_distance_near_unit_circle(
                b, a, minimum_magnitude));
    };
    const double distance = symmetric_near_unit_distance(
        cpu_spectrum.eigenvalues, device_spectrum.eigenvalues);
    const double cpu_reference_distance = symmetric_near_unit_distance(
        cpu_spectrum.eigenvalues, reference_spectrum.eigenvalues);
    const double device_reference_distance = symmetric_near_unit_distance(
        device_spectrum.eigenvalues, reference_spectrum.eigenvalues);

    printf("SYCL/CPU dense spectrum block (0,3), %d steps: "
           "float-distance=%e to-double=%e/%e residuals=%e/%e "
           "leakage=%e/%e\n",
           operator_steps, distance, cpu_reference_distance,
           device_reference_distance,
           cpu_spectrum.max_right_residual,
           device_spectrum.max_right_residual,
           cpu_spectrum.max_fourier_leakage,
           device_spectrum.max_fourier_leakage);
    print_largest_eigenvalues("CPU double", reference_spectrum.eigenvalues);
    print_largest_eigenvalues("CPU float", cpu_spectrum.eigenvalues);
    print_largest_eigenvalues("SYCL float", device_spectrum.eigenvalues);
    assert_int_equal(device_spectrum.block_size, cpu_spectrum.block_size);
    assert_true(distance < 1e-3);
    assert_true(device_reference_distance < 1e-3);
    assert_true(device_spectrum.max_right_residual < 1e-4);
    assert_true(device_spectrum.max_left_residual < 1e-4);
    assert_true(device_spectrum.max_fourier_leakage < 5e-5);
}

void test_sycl_critical_block_detects_instability(void**) {
    Config re100_config = make_re100_n16_config();
    constexpr int screening_steps = 1;
    fdm::NSCylFourierBlockReference<float, true> re100_cpu(
        re100_config, 2, 3, screening_steps);
    fdm::NSCylFourierBlockReference<double, true> re100_reference(
        re100_config, 2, 3, screening_steps);
    fdm::NSCylSyclFourierBlockReference<float> re100_device(
        queue(), re100_config, 2, 3, screening_steps);
    const auto re100_cpu_spectrum = fdm::solve_ns_cyl_dense_block(
        re100_cpu, kDt, 0.0, 1e-4);
    const auto re100_reference_spectrum = fdm::solve_ns_cyl_dense_block(
        re100_reference, kDt, 0.0, 1e-10);
    const auto re100_device_spectrum = fdm::solve_ns_cyl_dense_block(
        re100_device, kDt, 0.0, 1e-4);

    auto summarize = [](const auto& spectrum) {
        int outside = 0;
        double largest = 0;
        for (const auto& value : spectrum.eigenvalues) {
            const double magnitude = std::abs(value);
            outside += magnitude > 1.0;
            largest = std::max(largest, magnitude);
        }
        return std::pair{outside,
            std::log(largest)/spectrum.operator_steps/kDt};
    };
    const auto [cpu_outside, cpu_growth] = summarize(re100_cpu_spectrum);
    const auto [reference_outside, reference_growth] =
        summarize(re100_reference_spectrum);
    const auto [device_outside, device_growth] =
        summarize(re100_device_spectrum);
    printf("Re=100 n=16 dense block (2,3), %d steps: "
           "outside unit circle CPU-float/double/SYCL=%d/%d/%d "
           "max growth=%+.6e/%+.6e/%+.6e\n",
           screening_steps, cpu_outside, reference_outside, device_outside,
           cpu_growth, reference_growth, device_growth);
    print_largest_eigenvalues(
        "Re=100 CPU double", re100_reference_spectrum.eigenvalues);
    print_largest_eigenvalues(
        "Re=100 SYCL float", re100_device_spectrum.eigenvalues);
    assert_int_equal(reference_outside, 4);
    assert_int_equal(cpu_outside, reference_outside);
    assert_int_equal(device_outside, reference_outside);
    assert_true(device_growth > 0);
    assert_true(std::abs(device_growth-reference_growth) < 3e-3);
    assert_true(re100_device_spectrum.max_right_residual < 1e-4);
    assert_true(re100_device_spectrum.max_left_residual < 1e-4);
    assert_true(re100_device_spectrum.max_fourier_leakage < 5e-5);
}

// Divergence of cell (i,k,j), evaluated in double from the float fields.
// The three differences are each O(|velocity|/spacing) and largely cancel, so
// their magnitude is what sets the float noise floor -- reported alongside.
struct Divergence {
    double value;
    double scale;
};

Divergence cell_divergence(NSCylSycl<float>& ns, int i, int k, int j) {
    auto u = ns.ua(), v = ns.va(), w = ns.wa();
    const double r = double(ns.r0)+double(ns.dr)*j-double(ns.dr)/2;
    const double radial =
        ((r+0.5*ns.dr)*u(i,k,j)-(r-0.5*ns.dr)*u(i,k,j-1))/(r*ns.dr);
    const double axial = (double(v(i,k,j))-v(i,k-1,j))/ns.dz;
    const double azimuthal = (double(w(i,k,j))-w(i-1,k,j))/(r*ns.dphi);
    return {radial+axial+azimuthal,
            std::max({std::abs(radial), std::abs(axial), std::abs(azimuthal)})};
}

// z is periodic here, so every axial face is an unknown and the projection
// must be exact in every axial plane.  Leaving the last face k=nz-1 out of
// the update -- as a range of nz-1 did -- shows up as a large divergence in
// the planes k=0 and k=nz-1 while the rest stay at round-off.
void test_sycl_projection_is_divergence_free(void**) {
    NSCylSycl<float> ns(queue(), kNr, kNz, kNphi, kR0, kR, kLz, kU0, kRe, kDt);
    fill_smooth_state(ns);

    ns.step();
    ns.step();
    queue().wait();

    double max_divergence = 0;
    double max_scale = 0;
    double worst_plane[kNz] = {};
    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = 0; k < ns.nz; ++k) {
            for (int j = 2; j < ns.nr; ++j) {
                const Divergence divergence = cell_divergence(ns, i, k, j);
                max_divergence = std::max(max_divergence, std::abs(divergence.value));
                max_scale = std::max(max_scale, divergence.scale);
                worst_plane[k] = std::max(worst_plane[k], std::abs(divergence.value));
            }
        }
    }

    printf("sycl float: max|div| = %e (term scale %e, relative %e)\n",
           max_divergence, max_scale, max_divergence/max_scale);
    printf("sycl float: max|div| in planes k=0 / k=nz-1 = %e / %e\n",
           worst_plane[0], worst_plane[kNz-1]);
    // Round-off only: single precision leaves ~1e-7 of the term magnitude.
    assert_true(max_divergence < 1e-5*max_scale);
}

// The radial pressure ghost is built from the complete intermediate radial
// momentum F and then handed to a Dirichlet solve, so it lags the solution by
// one step.  As on the CPU the residual is not arbitrary:
//
//     div|wall cell = -((r -+ dr/2)/r) * (dt/dr^2) * (p_new - p_old)
//
// Pinning this identity down proves the ghost really is p(1) - dr*F(0)/dt:
// the previous single-viscous-term formula does not satisfy it.
void test_sycl_radial_wall_divergence_matches_pressure_lag(void**) {
    NSCylSycl<float> ns(queue(), kNr, kNz, kNphi, kR0, kR, kLz, kU0, kRe, kDt);
    fill_smooth_state(ns);

    ns.step();
    queue().wait();

    auto p = ns.pa();
    std::vector<float> previous_p(ns.nphi*ns.nz*ns.nr);
    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = 0; k < ns.nz; ++k) {
            for (int j = 1; j <= ns.nr; ++j) {
                previous_p[(i*ns.nz+k)*ns.nr+j-1] = p(i,k,j);
            }
        }
    }

    ns.step();
    queue().wait();

    double max_identity_error = 0;
    double max_predicted = 0;
    double max_scale = 0;
    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = 0; k < ns.nz; ++k) {
            for (int j : {1, ns.nr}) {
                const double r = double(ns.r0)+double(ns.dr)*j-double(ns.dr)/2;
                const double face = (j == 1) ? r-double(ns.dr)/2 : r+double(ns.dr)/2;
                const double delta_p =
                    double(p(i,k,j))-previous_p[(i*ns.nz+k)*ns.nr+j-1];
                const double predicted =
                    -(face/r)*(double(ns.dt)/(double(ns.dr)*ns.dr))*delta_p;
                const Divergence divergence = cell_divergence(ns, i, k, j);
                max_predicted = std::max(max_predicted, std::abs(predicted));
                max_scale = std::max(max_scale, divergence.scale);
                max_identity_error = std::max(
                    max_identity_error, std::abs(divergence.value-predicted));
            }
        }
    }

    printf("sycl float: radial lag identity residual = %e "
           "(predicted magnitude %e, term scale %e)\n",
           max_identity_error, max_predicted, max_scale);
    assert_true(max_predicted > 1e-6);
    assert_true(max_identity_error < 1e-5*max_scale);
}

} // namespace

int main() {
    const CMUnitTest tests[] = {
        cmocka_unit_test(test_sycl_poisson_matches_cpu_float_reference),
        cmocka_unit_test(test_sycl_fourier_block_poisson_matches_full),
        cmocka_unit_test(test_sycl_step_matches_cpu_float_reference),
        cmocka_unit_test(test_sycl_linear_fourier_blocks_match_cpu),
        cmocka_unit_test(
            test_sycl_batched_blocks_match_individual_applications),
        cmocka_unit_test(
            test_sycl_linear_step_matches_centered_nonlinear_difference),
        cmocka_unit_test(test_sycl_linear_block_dense_spectrum_matches_cpu),
        cmocka_unit_test(test_sycl_critical_block_detects_instability),
        cmocka_unit_test(test_sycl_projection_is_divergence_free),
        cmocka_unit_test(test_sycl_radial_wall_divergence_matches_pressure_lag),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
