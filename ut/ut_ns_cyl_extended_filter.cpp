#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "config.h"
#include "ns_cyl_extended_filter.h"
#include "ns_cyl_fourier_block.h"
#include "ns_cyl_spectral_modes.h"
#include "ns_cyl_spectral_projector.h"
#include "ns_cyl_state.h"

extern "C" {
#include <cmocka.h>
}

namespace {

using T = double;
using Task = fdm::NSCyl<T, true, fdm::tensor_flag::periodic>;
using Layout = fdm::NSCylStateLayout<T>;
using Component = Layout::Component;
using Filter = fdm::NSCylExtendedSpectralFilter<T>;

Config make_config(double base_outer_radius=2.0) {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl_extended_filter",
        "--ns:r=1.0",
        "--ns:R=3.0",
        "--ns:h1=0.0",
        "--ns:h2=6.283185307179586",
        "--ns:nr=8",
        "--ns:nz=4",
        "--ns:nphi=4",
        "--ns:u0=1.0",
        "--ns:Re=20.0",
        "--ns:dt=0.001",
        "--ns:verbose=0",
        "--spectral:base_outer_radius="+std::to_string(base_outer_radius),
        "--extended:response_condition_limit=1e12"
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    config.rewrite(static_cast<int>(argv.size()), argv.data());
    return config;
}

fdm::NSCylSpectralProjector<T> make_projector(const Config& config) {
    fdm::NSCylFourierBlockReference<T, true> block(config, 0, 1);
    const Layout layout(8, 4, 4);
    const int phase = 0;
    const int omega_coordinate = phase*layout.radial_size
        +layout.radial_index(Component::v, 5);
    const int original_coordinate = phase*layout.radial_size
        +layout.radial_index(Component::v, 2);

    fdm::NSCylSpectralMode<T> mode;
    mode.m = block.m();
    mode.l = block.l();
    mode.phase_count = block.phase_count();
    mode.radial_size = block.radial_size();
    mode.block_size = block.size();
    mode.pressure_gauge_fixed = block.pressure_gauge_fixed();
    mode.multiplier = {1.01, 0.0};
    mode.growth_rate = 1.0;
    mode.frequency = 0;
    mode.right_residual = 0;
    mode.left_residual = 0;
    mode.growing = true;
    mode.residual_accepted = true;
    mode.column_count = 1;
    mode.right_columns.assign(block.size(), T(0));
    mode.left_columns.assign(block.size(), T(0));
    mode.right_columns[omega_coordinate] = T(1);
    mode.left_columns[omega_coordinate] = T(1);
    mode.left_columns[original_coordinate] = T(1);

    fdm::NSCylSpectralModeSet<T> modes;
    modes.append_filterable_mode(std::move(mode));
    return fdm::NSCylSpectralProjector<T>(modes, 1e6);
}

int packed_index(const Layout& layout, Component component,
                 int i, int k, int j) {
    int offset = 0;
    int radial_size = layout.nr;
    switch (component) {
    case Component::u:
        offset = layout.u_offset;
        radial_size = layout.nr-1;
        break;
    case Component::v:
        offset = layout.v_offset;
        break;
    case Component::w:
        offset = layout.w_offset;
        break;
    case Component::p:
        offset = layout.p_offset;
        break;
    }
    return offset+(i*layout.nz+k)*radial_size+(j-1);
}

double maximum_divergence(const Config& config,
                          const std::vector<T>& packed) {
    Task state(config);
    const Layout layout(state);
    layout.unpack(state, packed.data());
    double result = 0;
    for (int i = 0; i < state.nphi; ++i) {
        for (int k = 0; k < state.nz; ++k) {
            for (int j = 1; j <= state.nr; ++j) {
                const double radius = state.r0+(j-0.5)*state.dr;
                const double divergence =
                    ((radius+0.5*state.dr)*state.u[i][k][j]
                     -(radius-0.5*state.dr)*state.u[i][k][j-1])
                        /(radius*state.dr)
                    +(state.v[i][k][j]-state.v[i][k-1][j])/state.dz
                    +(state.w[i][k][j]-state.w[i-1][k][j])
                        /(radius*state.dphi);
                result = std::max(result, std::abs(divergence));
            }
        }
    }
    return result;
}

void test_biorthogonal_auxiliary_correction(void**) {
    Config config = make_config();
    auto projector = make_projector(config);
    Filter filter(config, std::move(projector));
    const Layout layout(8, 4, 4);
    fdm::PeriodicPackedFFT2<T> fft(layout.nphi, layout.nz);

    std::vector<T> state(layout.state_size, T(0));
    std::vector<T> values(fft.size());
    std::vector<T> plane(fft.size(), T(0));
    plane[1] = T(1);
    fft.synthesis(plane.data(), values.data());
    for (int i = 0; i < layout.nphi; ++i) {
        for (int k = 0; k < layout.nz; ++k) {
            state[packed_index(layout, Component::v, i, k, 2)] =
                values[static_cast<std::size_t>(i)*layout.nz+k];
        }
    }
    const auto before = state;
    const auto diagnostics = filter.apply(state);
    const auto boundary = filter.correction_boundary_velocity();

    std::vector<T> correction(state.size());
    for (std::size_t index = 0; index < correction.size(); ++index) {
        correction[index] = state[index]-before[index];
    }
    assert_true(diagnostics.unstable_coordinate_norm_before > 0.5);
    assert_true(diagnostics.unstable_coordinate_norm_after
                < 1e-12*diagnostics.unstable_coordinate_norm_before);
    assert_true(diagnostics.correction_velocity_norm > 0);
    assert_true(diagnostics.original_domain_change_norm < 1e-13);
    assert_true(maximum_divergence(config, correction) < 1e-11);
    assert_int_equal(boundary.nphi, layout.nphi);
    assert_int_equal(boundary.nz, layout.nz);
    assert_true(boundary.rms_norm() > 0);
    for (int i = 0; i < layout.nphi; ++i) {
        for (int k = 0; k < layout.nz; ++k) {
            const std::size_t plane = static_cast<std::size_t>(i)*layout.nz+k;
            assert_float_equal(boundary.radial[plane], 0.0, 1e-14);
            assert_float_equal(
                boundary.axial[plane],
                T(0.5)*correction[packed_index(
                    layout, Component::v, i, k, 5)], 1e-14);
            assert_float_equal(
                boundary.azimuthal[plane],
                T(0.5)*correction[packed_index(
                    layout, Component::w, i, k, 5)], 1e-14);
        }
    }
}

void test_supported_correction_can_target_nonzero_coordinates(void**) {
    Config config = make_config();
    auto projector = make_projector(config);
    Filter filter(config, std::move(projector));
    const Layout layout(8, 4, 4);
    fdm::PeriodicPackedFFT2<T> fft(layout.nphi, layout.nz);

    std::vector<T> state(layout.state_size, T(0));
    std::vector<T> values(fft.size());
    std::vector<T> plane(fft.size(), T(0));
    plane[1] = T(1);
    fft.synthesis(plane.data(), values.data());
    for (int i = 0; i < layout.nphi; ++i) {
        for (int k = 0; k < layout.nz; ++k) {
            state[packed_index(layout, Component::v, i, k, 2)] =
                values[static_cast<std::size_t>(i)*layout.nz+k];
        }
    }
    const auto before = state;
    std::vector<T> target(state.size());
    for (std::size_t index = 0; index < state.size(); ++index) {
        target[index] = T(0.25)*state[index];
    }

    const auto diagnostics = filter.apply_towards(state, target);
    assert_true(diagnostics.unstable_coordinate_norm_before > 0.25);
    assert_true(diagnostics.unstable_coordinate_norm_after
                < 1e-12*diagnostics.unstable_coordinate_norm_before);

    // Only the auxiliary annulus may change, even though the requested
    // target has different values in Omega.
    for (int i = 0; i < layout.nphi; ++i) {
        for (int k = 0; k < layout.nz; ++k) {
            for (int j = 1; j < 4; ++j) {
                assert_float_equal(
                    state[packed_index(layout, Component::u, i, k, j)],
                    before[packed_index(layout, Component::u, i, k, j)],
                    1e-14);
            }
            for (Component component : {Component::v, Component::w}) {
                for (int j = 1; j <= 4; ++j) {
                    assert_float_equal(
                        state[packed_index(layout, component, i, k, j)],
                        before[packed_index(layout, component, i, k, j)],
                        1e-14);
                }
            }
        }
    }
    assert_true(filter.correction_boundary_velocity().rms_norm() > 0);
}

void test_auxiliary_interface_must_be_grid_aligned(void**) {
    Config config = make_config(2.01);
    auto projector = make_projector(make_config());
    bool rejected = false;
    try {
        Filter filter(config, std::move(projector));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert_true(rejected);
}

} // namespace

int main() {
    const CMUnitTest tests[] = {
        cmocka_unit_test(test_biorthogonal_auxiliary_correction),
        cmocka_unit_test(test_supported_correction_can_target_nonzero_coordinates),
        cmocka_unit_test(test_auxiliary_interface_must_be_grid_aligned)
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
