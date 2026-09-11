#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "config.h"
#include "ns_cyl_extended_filter.h"
#include "ns_cyl_fourier_block.h"
#include "ns_cyl_spectral_filter.h"
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

Config make_config(double base_outer_radius=2.0,
                   double response_regularization=0.0,
                   int response_basis_count=1,
                   int response_trace_horizon_steps=0,
                   int response_trace_sample_stride=1,
                   double response_cost_ridge=0.0) {
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
        "--extended:response_condition_limit=1e12",
        "--extended:response_regularization="
            +std::to_string(response_regularization),
        "--extended:response_basis_count="
            +std::to_string(response_basis_count),
        "--extended:response_trace_horizon_steps="
            +std::to_string(response_trace_horizon_steps),
        "--extended:response_trace_sample_stride="
            +std::to_string(response_trace_sample_stride),
        "--extended:response_cost_ridge="
            +std::to_string(response_cost_ridge)
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    config.rewrite(static_cast<int>(argv.size()), argv.data());
    return config;
}

fdm::NSCylSpectralProjector<T> make_projector(
    const Config& config, bool extended_radial_profile=false) {
    fdm::NSCylFourierBlockReference<T, true> block(config, 0, 1);
    const Layout layout(8, 4, 4);
    const int phase = 0;
    const int omega_coordinate = phase*layout.radial_size
        +layout.radial_index(Component::v, 5);
    const int second_omega_coordinate = phase*layout.radial_size
        +layout.radial_index(Component::v, 6);
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
    if (extended_radial_profile) {
        mode.right_columns[second_omega_coordinate] = T(0.5);
    }
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
    const auto diagnostics = filter.apply(state, true);
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
    assert_int_equal(diagnostics.blocks.size(), 1);
    const auto& block = diagnostics.blocks.front();
    assert_true(std::abs(block.correction_velocity_norm
                         -diagnostics.correction_velocity_norm)
                < 1e-12*diagnostics.correction_velocity_norm);
    assert_true(std::abs(block.boundary_rms-boundary.rms_norm())
                < 1e-12*boundary.rms_norm());
    assert_true(std::abs(block.boundary_maximum-boundary.maximum_norm())
                < 1e-12*boundary.maximum_norm());
    assert_true(std::abs(block.coordinate_to_correction_gain
                         -block.correction_velocity_norm
                             /block.unstable_coordinate_norm_before)
                < 1e-12*block.coordinate_to_correction_gain);
    assert_true(std::abs(block.coordinate_to_boundary_rms_gain
                         -block.boundary_rms
                             /block.unstable_coordinate_norm_before)
                < 1e-12*block.coordinate_to_boundary_rms_gain);
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

void test_regularization_trades_residual_for_boundary_energy(void**) {
    const double regularization = 0.1;
    const Layout layout(8, 4, 4);
    fdm::PeriodicPackedFFT2<T> fft(layout.nphi, layout.nz);
    std::vector<T> input(layout.state_size, T(0));
    std::vector<T> values(fft.size());
    std::vector<T> plane(fft.size(), T(0));
    plane[1] = T(1);
    fft.synthesis(plane.data(), values.data());
    for (int i = 0; i < layout.nphi; ++i) {
        for (int k = 0; k < layout.nz; ++k) {
            input[packed_index(layout, Component::v, i, k, 2)] =
                values[static_cast<std::size_t>(i)*layout.nz+k];
        }
    }

    auto exact_state = input;
    Filter exact_filter(make_config(), make_projector(make_config()));
    const auto exact = exact_filter.apply(exact_state, true);

    Config regularized_config = make_config(2.0, regularization);
    auto regularized_state = input;
    Filter regularized_filter(
        regularized_config, make_projector(regularized_config));
    const auto regularized_result = regularized_filter.apply(
        regularized_state, true);

    const auto& exact_block = exact.blocks.front();
    const auto& regularized_block = regularized_result.blocks.front();
    const double boundary_gain = exact_block.boundary_rms
        /exact_block.unstable_coordinate_norm_before;
    const double expected_scale = 1/(1+regularization
        *boundary_gain*boundary_gain);
    const double observed_scale = regularized_block.boundary_rms
        /exact_block.boundary_rms;
    const double residual_ratio =
        regularized_block.unstable_coordinate_norm_after
        /regularized_block.unstable_coordinate_norm_before;

    assert_float_equal(observed_scale, expected_scale, 1e-12);
    assert_float_equal(residual_ratio, 1-expected_scale, 1e-12);
    assert_true(regularized_result.correction_velocity_norm
                < exact.correction_velocity_norm);
    assert_true(regularized_filter.correction_boundary_velocity().rms_norm()
                < exact_filter.correction_boundary_velocity().rms_norm());
    assert_true(regularized_result.original_domain_change_norm < 1e-13);
    std::vector<T> correction(regularized_state.size());
    for (std::size_t index = 0; index < correction.size(); ++index) {
        correction[index] = regularized_state[index]-input[index];
    }
    assert_true(maximum_divergence(regularized_config, correction) < 1e-11);
}

void test_expanded_continuation_minimizes_boundary_trace(void**) {
    const Layout layout(8, 4, 4);
    fdm::PeriodicPackedFFT2<T> fft(layout.nphi, layout.nz);
    std::vector<T> input(layout.state_size, T(0));
    std::vector<T> values(fft.size());
    std::vector<T> plane(fft.size(), T(0));
    plane[1] = T(1);
    fft.synthesis(plane.data(), values.data());
    for (int i = 0; i < layout.nphi; ++i) {
        for (int k = 0; k < layout.nz; ++k) {
            input[packed_index(layout, Component::v, i, k, 2)] =
                values[static_cast<std::size_t>(i)*layout.nz+k];
        }
    }

    Config exact_config = make_config();
    auto exact_state = input;
    Filter exact_filter(
        exact_config, make_projector(exact_config, true));
    const auto exact = exact_filter.apply(exact_state, true);

    Config expanded_config = make_config(2.0, 0.0, 2, 0, 1, 1e-12);
    auto expanded_state = input;
    Filter expanded_filter(
        expanded_config, make_projector(expanded_config, true));
    const auto expanded = expanded_filter.apply(expanded_state, true);

    assert_int_equal(exact.blocks.front().continuation_dimension, 1);
    assert_int_equal(expanded.blocks.front().continuation_dimension, 2);
    assert_true(expanded.unstable_coordinate_norm_after
                < 1e-11*expanded.unstable_coordinate_norm_before);
    assert_true(expanded.original_domain_change_norm < 1e-13);
    assert_true(expanded.blocks.front().boundary_rms
                <= exact.blocks.front().boundary_rms*(1+1e-9));
    std::vector<T> correction(expanded_state.size());
    for (std::size_t index = 0; index < correction.size(); ++index) {
        correction[index] = expanded_state[index]-input[index];
    }
    assert_true(maximum_divergence(expanded_config, correction) < 1e-11);

    // Exercise the time-averaged trace path as well.  Its optimum need not
    // minimize the initial trace, but it must retain the exact modal
    // constraint and support restriction.
    Config horizon_config = make_config(2.0, 0.0, 2, 4, 2, 1e-10);
    auto horizon_state = input;
    Filter horizon_filter(
        horizon_config, make_projector(horizon_config, true));
    const auto horizon = horizon_filter.apply(horizon_state, true);
    assert_true(horizon.unstable_coordinate_norm_after
                < 1e-10*horizon.unstable_coordinate_norm_before);
    assert_true(horizon.original_domain_change_norm < 1e-13);
    for (std::size_t index = 0; index < correction.size(); ++index) {
        correction[index] = horizon_state[index]-input[index];
    }
    assert_true(maximum_divergence(horizon_config, correction) < 1e-11);
}

void test_zero_order_nonlinear_target_matches_linear_correction(void**) {
    Config config = make_config();
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

    Task geometry(config);
    std::vector<T> reference(layout.state_size, T(0));
    auto stable_target = state;
    fdm::NSCylSpectralFilter<T> full_filter(
        layout.nr, layout.nphi, layout.nz, make_projector(config));
    full_filter.remove_packed(geometry, stable_target, reference);

    auto linear_state = state;
    Filter linear_filter(config, make_projector(config));
    linear_filter.apply(linear_state);
    const auto linear_boundary =
        linear_filter.correction_boundary_velocity();

    auto target_state = state;
    Filter target_filter(config, make_projector(config));
    const auto diagnostics = target_filter.apply_towards(
        target_state, stable_target);
    const auto target_boundary =
        target_filter.correction_boundary_velocity();

    assert_true(diagnostics.unstable_coordinate_norm_after < 1e-12);
    for (std::size_t index = 0; index < state.size(); ++index) {
        assert_float_equal(target_state[index], linear_state[index], 1e-12);
    }
    for (std::size_t index = 0;
         index < linear_boundary.radial.size(); ++index) {
        assert_float_equal(
            target_boundary.radial[index], linear_boundary.radial[index],
            1e-12);
        assert_float_equal(
            target_boundary.axial[index], linear_boundary.axial[index],
            1e-12);
        assert_float_equal(
            target_boundary.azimuthal[index],
            linear_boundary.azimuthal[index], 1e-12);
    }
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
        cmocka_unit_test(test_regularization_trades_residual_for_boundary_energy),
        cmocka_unit_test(test_expanded_continuation_minimizes_boundary_trace),
        cmocka_unit_test(test_zero_order_nonlinear_target_matches_linear_correction),
        cmocka_unit_test(test_auxiliary_interface_must_be_grid_aligned)
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
