// test_ns_cube_sycl.cpp
// Compare NSCubeSycl (SYCL/GPU) against NSCube (CPU) field-by-field.
// Both use float; tolerance 0.1% relative accounts for FP reordering.

#include <sycl/sycl.hpp>
#include "ns_cube_sycl.h"
#include "ns_cube.h"
#include "config.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>

int main()
{
    constexpr int   N     = 8;      // small grid, runs fast
    constexpr int   STEPS = 5;
    constexpr float Re    = 10.f;
    constexpr float dt    = 0.001f;
    constexpr float U0    = 1.f;

    // ── CPU reference (NSCube<float>) ─────────────────────────────────────────
    Config cfg;
    {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "[ns]\nnx=%d\nnz=%d\nRe=%f\ndt=%f\nu0=%f\n",
            N, N, (double)Re, (double)dt, (double)U0);
        FILE* f = ::fmemopen(buf, std::strlen(buf), "r");
        cfg.load(f);
        ::fclose(f);
    }
    fdm::NSCube<float, false> cpu(cfg);

    // ── SYCL (NSCubeSycl<float>) ──────────────────────────────────────────────
    sycl::queue q{
        []() {
            for (auto& p : sycl::platform::get_platforms())
                for (auto& d : p.get_devices())
                    if (d.is_gpu()) return d;
            return sycl::device{sycl::cpu_selector_v};
        }(),
        sycl::property::queue::in_order{}};

    std::cout << "SYCL device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    const float lx = float(2 * M_PI);
    fdm::NSCubeSycl<float> gpu(q, N, N, N, lx, lx, lx, U0, Re, dt);

    // ── Run both for STEPS steps ──────────────────────────────────────────────
    for (int s = 0; s < STEPS; s++) {
        cpu.step();
        gpu.step();
    }

    // ── Field comparison helper ───────────────────────────────────────────────
    bool all_ok = true;

    auto check = [&](const char* name, auto fn) {
        double max_abs = 0, max_ref = 0;
        fn(max_abs, max_ref);
        double rel = max_ref > 1e-30 ? max_abs / max_ref : max_abs;
        bool ok = rel < 1e-3;
        std::printf("  %-4s  max_abs=%.3e  max_ref=%.3e  rel=%.3e  %s\n",
                    name, max_abs, max_ref, rel, ok ? "OK" : "FAIL");
        if (!ok) all_ok = false;
    };

    std::cout << "After " << STEPS << " steps (N=" << N << ", Re=" << Re << "):\n";

    check("u", [&](double& me, double& mr) {
        for (int i=1;i<=N;i++) for (int k=1;k<=N;k++) for (int j=1;j<=N-1;j++) {
            me = std::max(me, std::abs((double)gpu.u[i][k][j] - (double)cpu.u[i][k][j]));
            mr = std::max(mr, std::abs((double)cpu.u[i][k][j]));
        }
    });
    check("v", [&](double& me, double& mr) {
        for (int i=1;i<=N;i++) for (int k=1;k<=N-1;k++) for (int j=1;j<=N;j++) {
            me = std::max(me, std::abs((double)gpu.v[i][k][j] - (double)cpu.v[i][k][j]));
            mr = std::max(mr, std::abs((double)cpu.v[i][k][j]));
        }
    });
    check("w", [&](double& me, double& mr) {
        for (int i=1;i<=N-1;i++) for (int k=1;k<=N;k++) for (int j=1;j<=N;j++) {
            me = std::max(me, std::abs((double)gpu.w[i][k][j] - (double)cpu.w[i][k][j]));
            mr = std::max(mr, std::abs((double)cpu.w[i][k][j]));
        }
    });
    check("p", [&](double& me, double& mr) {
        for (int i=1;i<=N;i++) for (int k=1;k<=N;k++) for (int j=1;j<=N;j++) {
            me = std::max(me, std::abs((double)gpu.p[i][k][j] - (double)cpu.p[i][k][j]));
            mr = std::max(mr, std::abs((double)cpu.p[i][k][j]));
        }
    });

    std::cout << (all_ok ? "PASS\n" : "FAIL\n");
    return all_ok ? 0 : 1;
}
