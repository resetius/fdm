#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

#include "config.h"
#include "ns_cyl.h"

extern "C" {
#include <cmocka.h>
}

using fdm::NSCyl;
using fdm::tensor_flag;

namespace {

Config make_config(int nr, int nz, int nphi, bool random_v = false,
                   double u0 = 1.0, double reynolds = 10.0,
                   double dt = 1e-4) {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl",
        "--ns:r=1.0",
        "--ns:R=2.0",
        "--ns:h1=0.0",
        "--ns:h2=6.283185307179586",
        "--ns:nr=" + std::to_string(nr),
        "--ns:nz=" + std::to_string(nz),
        "--ns:nphi=" + std::to_string(nphi),
        "--ns:u0=" + std::to_string(u0),
        "--ns:Re=" + std::to_string(reynolds),
        "--ns:dt=" + std::to_string(dt),
        "--ns:vrandom=" + std::to_string(random_v ? 1 : 0),
        "--ns:verbose=0"
    };
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    config.rewrite(static_cast<int>(argv.size()), argv.data());
    return config;
}

template<typename T>
void check_discrete_couette_projection() {
    using Task = NSCyl<T, true, tensor_flag::periodic>;
    Config config = make_config(10, 8, 8);
    Task ns(config);

    // Давление точно компенсирует дискретный центробежный член w^2/r.
    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = 0; k < ns.nz; ++k) {
            for (int j = 1; j <= ns.nr; ++j) {
                const double r = ns.r0+(j-0.5)*ns.dr;
                ns.w[i][k][j] = static_cast<T>(ns.couette_velocity(r));
            }
            ns.p[i][k][1] = T(0);
            for (int j = 1; j < ns.nr; ++j) {
                const double r_face = ns.r0+j*ns.dr;
                const double w_face = 0.5*(ns.w[i][k][j]+ns.w[i][k][j+1]);
                ns.p[i][k][j+1] = static_cast<T>(
                    ns.p[i][k][j]+ns.dr*w_face*w_face/r_face);
            }
        }
    }

    ns.step();

    double max_divergence = 0;
    double max_radial_velocity = 0;
    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = 0; k < ns.nz; ++k) {
            assert_true(ns.u[i][k][0] == T(0));
            assert_true(ns.u[i][k][ns.nr] == T(0));
            for (int j = 1; j <= ns.nr; ++j) {
                const double r = ns.r0+(j-0.5)*ns.dr;
                const double divergence =
                    ((r+0.5*ns.dr)*ns.u[i][k][j]
                     -(r-0.5*ns.dr)*ns.u[i][k][j-1])/(r*ns.dr)
                    +(ns.v[i][k][j]-ns.v[i][k-1][j])/ns.dz
                    +(ns.w[i][k][j]-ns.w[i-1][k][j])/(r*ns.dphi);
                max_divergence = std::max(max_divergence, std::abs(divergence));
            }
            for (int j = 0; j <= ns.nr; ++j) {
                max_radial_velocity = std::max(
                    max_radial_velocity, std::abs(static_cast<double>(ns.u[i][k][j])));
            }
        }
    }

    const double tolerance = std::is_same_v<T, double> ? 2e-11 : 2e-5;
    assert_true(max_divergence < tolerance);
    assert_true(max_radial_velocity < tolerance);
}

void test_discrete_couette_projection_double(void**) {
    check_discrete_couette_projection<double>();
}

void test_discrete_couette_projection_float(void**) {
    check_discrete_couette_projection<float>();
}

void test_couette_profile_is_azimuthal_velocity(void**) {
    Config config = make_config(4, 4, 8, false, 1.25);
    NSCyl<double, true, tensor_flag::periodic> ns(config);
    assert_float_equal(ns.couette_velocity(ns.r0), ns.U0, 1e-14);
    assert_float_equal(ns.couette_velocity(ns.R), 0.0, 1e-14);
}

void test_nonperiodic_random_v_respects_staggered_walls(void**) {
    // nz > nr выявляет подмену радиального предела осевым.
    Config config = make_config(4, 7, 8, true);
    NSCyl<double, true, tensor_flag::none> ns(config);

    for (int i = 0; i < ns.nphi; ++i) {
        for (int j = 1; j <= ns.nr; ++j) {
            assert_true(ns.v[i][0][j] == 0.0);
            assert_true(ns.v[i][ns.nz][j] == 0.0);
        }
    }

    ns.step();

    for (int i = 0; i < ns.nphi; ++i) {
        for (int j = 1; j <= ns.nr; ++j) {
            assert_true(ns.v[i][0][j] == 0.0);
            assert_true(ns.v[i][ns.nz][j] == 0.0);
        }
    }
    for (const auto* field : {&ns.u, &ns.v, &ns.w, &ns.p}) {
        for (int index = 0; index < field->size; ++index) {
            assert_true(std::isfinite(field->vec[index]));
        }
    }
}

void test_linearized_step_matches_central_difference(void**) {
    using Task = NSCyl<double, true, tensor_flag::periodic>;
    constexpr double epsilon = 1e-4;
    Config config = make_config(4, 4, 8, false, 0.7, 7.0, 1e-3);
    Task plus(config);
    Task minus(config);
    Task linear(config);

    const auto base_u = [](int i, int k, int j, int nr) {
        return 0.08*std::sin(0.7*i+0.3*k)*std::sin(M_PI*j/nr);
    };
    const auto delta_u = [](int i, int k, int j, int nr) {
        return 0.05*std::cos(0.4*i-0.2*k)*std::sin(M_PI*j/nr);
    };
    const auto base_v = [](int i, int k, int j) {
        return 0.06*std::cos(0.3*i+0.5*k+0.2*j);
    };
    const auto delta_v = [](int i, int k, int j) {
        return 0.04*std::sin(0.5*i-0.3*k+0.1*j);
    };
    const auto base_w = [](int i, int k, int j) {
        return 0.3+0.07*std::sin(0.6*i+0.2*k+0.15*j);
    };
    const auto delta_w = [](int i, int k, int j) {
        return 0.03*std::cos(0.2*i-0.4*k+0.3*j);
    };
    const auto base_p = [](int i, int k, int j) {
        return 0.02*std::sin(0.2*i+0.3*k+0.4*j);
    };
    const auto delta_p = [](int i, int k, int j) {
        return 0.01*std::cos(0.3*i-0.2*k+0.5*j);
    };

    for (int i = 0; i < linear.nphi; ++i) {
        for (int k = 0; k < linear.nz; ++k) {
            for (int j = 0; j <= linear.nr; ++j) {
                const double base = (j == 0 || j == linear.nr)
                    ? 0.0 : base_u(i,k,j,linear.nr);
                const double delta = (j == 0 || j == linear.nr)
                    ? 0.0 : delta_u(i,k,j,linear.nr);
                plus.u[i][k][j] = base+epsilon*delta;
                minus.u[i][k][j] = base-epsilon*delta;
                linear.u[i][k][j] = delta;
                linear.u0[i][k][j] = base;
            }
            for (int j = 1; j <= linear.nr; ++j) {
                const double bv = base_v(i,k,j);
                const double dv = delta_v(i,k,j);
                const double bw = base_w(i,k,j);
                const double dw = delta_w(i,k,j);
                const double bp = base_p(i,k,j);
                const double dp = delta_p(i,k,j);
                plus.v[i][k][j] = bv+epsilon*dv;
                minus.v[i][k][j] = bv-epsilon*dv;
                linear.v[i][k][j] = dv;
                linear.v0[i][k][j] = bv;
                plus.w[i][k][j] = bw+epsilon*dw;
                minus.w[i][k][j] = bw-epsilon*dw;
                linear.w[i][k][j] = dw;
                linear.w0[i][k][j] = bw;
                plus.p[i][k][j] = bp+epsilon*dp;
                minus.p[i][k][j] = bp-epsilon*dp;
                linear.p[i][k][j] = dp;
            }

            linear.u0[i][k][-1] = linear.u0[i][k][1];
            linear.u0[i][k][linear.nr+1] = linear.u0[i][k][linear.nr-1];
            linear.v0[i][k][0] = -linear.v0[i][k][1];
            linear.v0[i][k][linear.nr+1] = -linear.v0[i][k][linear.nr];
            linear.w0[i][k][0] = 2*linear.U0-linear.w0[i][k][1];
            linear.w0[i][k][linear.nr+1] = -linear.w0[i][k][linear.nr];
        }
    }

    // Для возмущения U0=0; скорость стенки уже учтена в w0.
    linear.U0 = 0;
    plus.step();
    minus.step();
    linear.L_step();

    double max_error = 0;
    double max_reference = 0;
    const auto compare = [&](double actual, double positive, double negative) {
        const double reference = (positive-negative)/(2*epsilon);
        max_error = std::max(max_error, std::abs(actual-reference));
        max_reference = std::max(max_reference, std::abs(reference));
    };

    for (int i = 0; i < linear.nphi; ++i) {
        for (int k = 0; k < linear.nz; ++k) {
            for (int j = 1; j < linear.nr; ++j) {
                compare(linear.u[i][k][j], plus.u[i][k][j], minus.u[i][k][j]);
            }
            for (int j = 1; j <= linear.nr; ++j) {
                compare(linear.v[i][k][j], plus.v[i][k][j], minus.v[i][k][j]);
                compare(linear.w[i][k][j], plus.w[i][k][j], minus.w[i][k][j]);
                compare(linear.p[i][k][j], plus.p[i][k][j], minus.p[i][k][j]);
            }
        }
    }

    assert_true(max_reference > 0);
    assert_true(max_error/max_reference < 2e-8);
}

// Дискретная дивергенция в обозначениях poisson().
template<typename Task>
double cell_divergence(Task& ns, int i, int k, int j) {
    const double r = ns.r0+ns.dr*j-ns.dr/2;
    return ((r+0.5*ns.dr)*ns.u[i][k][j]-(r-0.5*ns.dr)*ns.u[i][k][j-1])/(r*ns.dr)
        +(ns.v[i][k][j]-ns.v[i][k-1][j])/ns.dz
        +(ns.w[i][k][j]-ns.w[i-1][k][j])/(r*ns.dphi);
}

template<typename Task>
void fill_smooth_state(Task& ns) {
    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = ns.z1; k <= ns.zn; ++k) {
            for (int j = 1; j <= ns.nr; ++j) {
                const double r = ns.r0+(j-0.5)*ns.dr;
                ns.w[i][k][j] = ns.couette_velocity(r)
                    +0.05*std::cos(2*M_PI*i/ns.nphi)*std::sin(0.7*k);
                ns.p[i][k][j] = 0.02*std::sin(2*M_PI*i/ns.nphi+0.4*k);
            }
            for (int j = 1; j < ns.nr; ++j) {
                ns.u[i][k][j] =
                    0.03*std::sin(2*M_PI*i/ns.nphi)*std::sin(M_PI*j/ns.nr)
                    *std::cos(0.5*k);
            }
        }
        for (int k = ns.z1; k < ns.nz; ++k) {
            for (int j = 1; j <= ns.nr; ++j) {
                ns.v[i][k][j] =
                    0.02*std::cos(4*M_PI*i/ns.nphi)
                    *std::sin(M_PI*(j-0.5)/ns.nr)*std::sin(0.3*k);
            }
        }
    }
}

// Периодический тензор заворачивает индекс, поэтому границы проверяем явно.
void test_z_bounds_match_flag(void**) {
    Config config = make_config(4, 7, 8);
    NSCyl<double, true, tensor_flag::none> bounded(config);

    assert_int_equal(bounded.z_, -1);
    assert_int_equal(bounded.z0, 0);
    assert_int_equal(bounded.z1, 1);
    assert_int_equal(bounded.zn, bounded.nz);
    assert_int_equal(bounded.znn, bounded.nz+1);

    // v и G содержат обе торцевые грани.
    assert_int_equal(bounded.v.offsets[2], -1);
    assert_int_equal(bounded.v.offsets[3], bounded.nz+1);
    assert_int_equal(bounded.G.offsets[2], 0);
    assert_int_equal(bounded.G.offsets[3], bounded.nz);
    assert_int_equal(bounded.F.offsets[2], 1);
    assert_int_equal(bounded.F.offsets[3], bounded.nz);
    assert_int_equal(bounded.p.offsets[2], 0);
    assert_int_equal(bounded.p.offsets[3], bounded.nz+1);

    Config periodic_config = make_config(4, 8, 8);
    NSCyl<double, true, tensor_flag::periodic> periodic(periodic_config);

    assert_int_equal(periodic.z_, 0);
    assert_int_equal(periodic.z0, 0);
    assert_int_equal(periodic.z1, 0);
    assert_int_equal(periodic.zn, periodic.nz-1);
    assert_int_equal(periodic.znn, periodic.nz-1);

    for (const auto* offsets : {&periodic.u.offsets, &periodic.v.offsets,
                                &periodic.w.offsets, &periodic.p.offsets,
                                &periodic.G.offsets, &periodic.F.offsets}) {
        assert_int_equal((*offsets)[2], 0);
        assert_int_equal((*offsets)[3], periodic.nz-1);
    }
}

// Радиальные ячейки у стенки исключены из-за запаздывания давления.
void test_periodic_z_projection_is_divergence_free(void**) {
    using Task = NSCyl<double, true, tensor_flag::periodic>;
    Config config = make_config(8, 8, 8, false, 1.0, 10.0, 1e-4);
    Task ns(config);
    fill_smooth_state(ns);

    ns.step();
    ns.step();

    double max_divergence = 0;
    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = 0; k < ns.nz; ++k) {
            for (int j = 2; j < ns.nr; ++j) {
                max_divergence = std::max(
                    max_divergence, std::abs(cell_divergence(ns, i, k, j)));
            }
        }
    }
    printf("periodic z: max|div| = %e\n", max_divergence);
    assert_true(max_divergence < 1e-10);
}

// На торце граничное p использует значение с прошлого шага:
// div_wall = -(dt/dz^2)*(p_new-p_old).
void test_nonperiodic_z_wall_divergence_matches_pressure_lag(void**) {
    using Task = NSCyl<double, true, tensor_flag::none>;
    Config config = make_config(8, 7, 8, false, 1.0, 10.0, 1e-4);
    Task ns(config);
    fill_smooth_state(ns);

    ns.step();
    auto previous_p = ns.p;
    ns.step();

    const double factor = ns.dt/(ns.dz*ns.dz);
    double max_interior_divergence = 0;
    double max_wall_divergence = 0;
    double max_identity_error = 0;
    double max_predicted = 0;

    for (int i = 0; i < ns.nphi; ++i) {
        for (int j = 2; j < ns.nr; ++j) {
            for (int k = 2; k < ns.nz; ++k) {
                max_interior_divergence = std::max(
                    max_interior_divergence,
                    std::abs(cell_divergence(ns, i, k, j)));
            }
            for (int k : {1, ns.nz}) {
                const double divergence = cell_divergence(ns, i, k, j);
                const double predicted =
                    -factor*(ns.p[i][k][j]-previous_p[i][k][j]);
                max_wall_divergence = std::max(
                    max_wall_divergence, std::abs(divergence));
                max_predicted = std::max(max_predicted, std::abs(predicted));
                max_identity_error = std::max(
                    max_identity_error, std::abs(divergence-predicted));
            }
        }
    }

    printf("bounded z: max|div| interior = %e, at axial walls = %e\n",
           max_interior_divergence, max_wall_divergence);
    printf("bounded z: lag identity residual = %e (predicted magnitude %e)\n",
           max_identity_error, max_predicted);

    assert_true(max_interior_divergence < 1e-10);
    assert_true(max_predicted > 1e-12);
    assert_true(max_identity_error < 1e-10*std::max(1.0, max_predicted));
}

// Обработка торцов не должна выполняться при периодическом z.
void test_periodic_z_uniform_state_stays_uniform(void**) {
    using Task = NSCyl<double, true, tensor_flag::periodic>;
    Config config = make_config(8, 8, 8, false, 1.0, 10.0, 1e-4);
    Task ns(config);

    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = 0; k < ns.nz; ++k) {
            for (int j = 1; j <= ns.nr; ++j) {
                const double r = ns.r0+(j-0.5)*ns.dr;
                ns.w[i][k][j] = ns.couette_velocity(r)
                    +0.05*std::cos(2*M_PI*i/ns.nphi);
                ns.v[i][k][j] = 0.02*std::cos(4*M_PI*i/ns.nphi)
                    *std::sin(M_PI*(j-0.5)/ns.nr);
                ns.p[i][k][j] = 0.01*std::sin(2*M_PI*i/ns.nphi);
            }
            for (int j = 1; j < ns.nr; ++j) {
                ns.u[i][k][j] =
                    0.03*std::sin(2*M_PI*i/ns.nphi)*std::sin(M_PI*j/ns.nr);
            }
        }
    }

    ns.step();
    ns.step();

    double max_axial_variation = 0;
    for (int i = 0; i < ns.nphi; ++i) {
        for (int k = 1; k < ns.nz; ++k) {
            for (int j = 1; j <= ns.nr; ++j) {
                max_axial_variation = std::max({
                    max_axial_variation,
                    std::abs(ns.u[i][k][j]-ns.u[i][0][j]),
                    std::abs(ns.v[i][k][j]-ns.v[i][0][j]),
                    std::abs(ns.w[i][k][j]-ns.w[i][0][j]),
                    std::abs(ns.p[i][k][j]-ns.p[i][0][j])});
            }
        }
    }
    printf("periodic z: axial variation of a uniform state = %e\n",
           max_axial_variation);
    assert_true(max_axial_variation < 1e-12);
}

} // namespace

int main() {
    const CMUnitTest tests[] = {
        cmocka_unit_test(test_discrete_couette_projection_double),
        cmocka_unit_test(test_discrete_couette_projection_float),
        cmocka_unit_test(test_couette_profile_is_azimuthal_velocity),
        cmocka_unit_test(test_nonperiodic_random_v_respects_staggered_walls),
        cmocka_unit_test(test_linearized_step_matches_central_difference),
        cmocka_unit_test(test_z_bounds_match_flag),
        cmocka_unit_test(test_periodic_z_projection_is_divergence_free),
        cmocka_unit_test(test_nonperiodic_z_wall_divergence_matches_pressure_lag),
        cmocka_unit_test(test_periodic_z_uniform_state_stays_uniform),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
