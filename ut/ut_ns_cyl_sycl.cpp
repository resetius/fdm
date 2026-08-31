#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <sycl/sycl.hpp>

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
        cmocka_unit_test(test_sycl_projection_is_divergence_free),
        cmocka_unit_test(test_sycl_radial_wall_divergence_matches_pressure_lag),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
