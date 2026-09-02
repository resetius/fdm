#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "config.h"
#include "ns_cyl.h"
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

Config make_config() {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl_spectral_filter",
        "--ns:r=1.0",
        "--ns:R=2.0",
        "--ns:h1=0.0",
        "--ns:h2=6.283185307179586",
        "--ns:nr=4",
        "--ns:nz=4",
        "--ns:nphi=4",
        "--ns:u0=1.0",
        "--ns:Re=20.0",
        "--ns:dt=0.0001",
        "--ns:verbose=0"
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    config.rewrite(static_cast<int>(argv.size()), argv.data());
    return config;
}

Config make_unstable_config() {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl_spectral_filter",
        "--ns:r=1.5707963267948966",
        "--ns:R=3.141592653589793",
        "--ns:h1=0.0",
        "--ns:h2=10.0",
        "--ns:nr=8",
        "--ns:nz=8",
        "--ns:nphi=8",
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

fdm::NSCylSpectralMode<T> coordinate_mode(
    const fdm::NSCylFourierBlockReference<T, true>& block, int coordinate=0) {
    fdm::NSCylSpectralMode<T> mode;
    mode.m = block.m();
    mode.l = block.l();
    mode.phase_count = block.phase_count();
    mode.radial_size = block.radial_size();
    mode.block_size = block.size();
    mode.pressure_gauge_fixed = block.pressure_gauge_fixed();
    mode.multiplier = {1.1, 0.0};
    mode.growth_rate = 1.0;
    mode.frequency = 0;
    mode.right_residual = 0;
    mode.left_residual = 0;
    mode.growing = true;
    mode.residual_accepted = true;
    mode.column_count = 1;
    mode.right_columns.assign(block.size(), 0);
    mode.left_columns.assign(block.size(), 0);
    mode.right_columns.at(coordinate) = 1;
    mode.left_columns.at(coordinate) = 1;
    return mode;
}

fdm::NSCylSpectralProjector<T> coordinate_projector(
    const fdm::NSCylFourierBlockReference<T, true>& block) {
    fdm::NSCylSpectralModeSet<T> modes;
    modes.append_filterable_mode(coordinate_mode(block));
    return fdm::NSCylSpectralProjector<T>(modes, 1e6);
}

std::vector<T> couette_reference(Task& state, const Layout& layout) {
    layout.initialize_couette_state(state);
    return layout.pack(state);
}

std::vector<T> physical_block(
    fdm::NSCylFourierBlockReference<T, true>& block,
    const std::vector<T>& coefficients, const Layout& layout) {
    block.lift(coefficients.data());
    return layout.pack(block.task());
}

double relative_error(const std::vector<T>& actual,
                      const std::vector<T>& expected) {
    long double error = 0;
    long double scale = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const long double difference = actual[i]-expected[i];
        error += difference*difference;
        scale += static_cast<long double>(expected[i])*expected[i];
    }
    return std::sqrt(static_cast<double>(error/std::max(1.0L, scale)));
}

double vector_norm(const std::vector<T>& values) {
    long double result = 0;
    for (T value : values) {
        result += static_cast<long double>(value)*value;
    }
    return std::sqrt(static_cast<double>(result));
}

template<typename State>
double cell_divergence(State& state, int i, int k, int j) {
    const double radius = state.r0+(j-0.5)*state.dr;
    return ((radius+0.5*state.dr)*state.u[i][k][j]
            -(radius-0.5*state.dr)*state.u[i][k][j-1])
            /(radius*state.dr)
        +(state.v[i][k][j]-state.v[i][k-1][j])/state.dz
        +(state.w[i][k][j]-state.w[i-1][k][j])
            /(radius*state.dphi);
}

template<typename State>
double maximum_interior_divergence(State& state) {
    double result = 0;
    for (int i = 0; i < state.nphi; ++i) {
        for (int k = 0; k < state.nz; ++k) {
            for (int j = 2; j < state.nr; ++j) {
                result = std::max(
                    result, std::abs(cell_divergence(state, i, k, j)));
            }
        }
    }
    return result;
}

template<typename State>
void assert_finite_state(const State& state) {
    for (const auto* field : {&state.u, &state.v, &state.w, &state.p}) {
        for (int i = 0; i < field->size; ++i) {
            assert_true(std::isfinite(static_cast<double>(field->vec[i])));
        }
    }
}

template<typename State>
void assert_radial_wall_conditions(State& state, double tolerance) {
    for (int i = 0; i < state.nphi; ++i) {
        for (int k = 0; k < state.nz; ++k) {
            assert_true(std::abs(static_cast<double>(state.u[i][k][0]))
                        < tolerance);
            assert_true(std::abs(static_cast<double>(state.u[i][k][state.nr]))
                        < tolerance);
            assert_true(std::abs(0.5*(state.w[i][k][0]
                                      +state.w[i][k][1])-state.U0)
                        < tolerance);
            assert_true(std::abs(0.5*(state.w[i][k][state.nr]
                                      +state.w[i][k][state.nr+1]))
                        < tolerance);
            assert_true(std::abs(state.v[i][k][0]+state.v[i][k][1])
                        < tolerance);
            assert_true(std::abs(state.v[i][k][state.nr+1]
                                 +state.v[i][k][state.nr]) < tolerance);
        }
    }
}

void test_velocity_norm_uses_cylindrical_measure(void**) {
    const Config config = make_config();
    Task state(config);
    const Layout layout(state);
    std::vector<T> packed(layout.state_size, 0);
    std::fill(packed.begin()+layout.v_offset,
              packed.begin()+layout.w_offset, 1.0);
    std::fill(packed.begin()+layout.w_offset,
              packed.begin()+layout.p_offset, 1.0);
    std::fill(packed.begin()+layout.p_offset, packed.end(), 1e6);

    const double expected_squared = 2*M_PI*(state.h2-state.h1)
        *(state.R*state.R-state.r0*state.r0);
    const double actual = layout.velocity_norm(state, packed.data());
    assert_true(std::abs(actual*actual-expected_squared)
                < 2e-14*expected_squared);
}

void test_linear_unstable_trajectory_is_removed(void**) {
    const Config config = make_unstable_config();
    fdm::NSCylFourierBlockReference<T, true> block(config, 0, 3);
    const auto spectrum = fdm::solve_ns_cyl_dense_block(
        block, 0.001, 1e-8, 1e-10);
    const auto leading = std::max_element(
        spectrum.modes.begin(), spectrum.modes.end(),
        [](const auto& a, const auto& b) {
            return a.growth_rate < b.growth_rate;
        });
    assert_true(leading != spectrum.modes.end());
    assert_true(leading->growing);
    assert_int_equal(leading->column_count, 1);
    assert_true(leading->multiplier.real() > 1);
    assert_true(leading->multiplier.imag() == 0);

    fdm::NSCylSpectralModeSet<T> modes;
    modes.append_filterable_mode(*leading);
    const fdm::NSCylSpectralProjector<T> projector(modes, 1e6);
    const auto* block_projector = projector.find_block(0, 3);
    assert_non_null(block_projector);

    std::vector<T> state = leading->right_columns;
    std::vector<T> image(block.size());
    std::vector<T> unstable(block.size());
    const double initial_norm = vector_norm(state);
    double maximum_growth_error = 0;
    constexpr int trajectory_steps = 32;
    for (int step = 0; step <= trajectory_steps; ++step) {
        block_projector->project(unstable.data(), state.data());
        const double expected = initial_norm*std::pow(
            leading->multiplier.real(), step);
        maximum_growth_error = std::max(
            maximum_growth_error,
            std::abs(vector_norm(unstable)-expected)/expected);
        if (step != trajectory_steps) {
            block.apply(image.data(), state.data());
            state.swap(image);
        }
    }
    assert_true(maximum_growth_error < 2e-12);

    const double before = vector_norm(state);
    block_projector->remove(image.data(), state.data());
    block_projector->project(unstable.data(), image.data());
    const double immediate_ratio = vector_norm(unstable)/before;
    block.apply(state.data(), image.data());
    block_projector->project(unstable.data(), state.data());
    const double next_step_ratio = vector_norm(unstable)/before;
    printf("linear filter trajectory: growth_error=%e immediate=%e next=%e\n",
           maximum_growth_error, immediate_ratio, next_step_ratio);
    assert_true(immediate_ratio < 2e-13);
    assert_true(next_step_ratio < 2e-12);
}

void test_removes_selected_mode_and_preserves_complement(void**) {
    const Config config = make_config();
    Task state(config);
    const Layout layout(state);
    const auto reference = couette_reference(state, layout);
    fdm::NSCylFourierBlockReference<T, true> block(config, 0, 1);

    std::vector<T> block_state(block.size(), 0);
    block_state[0] = 2.0;
    block_state[1] = -0.75;
    const auto perturbation = physical_block(block, block_state, layout);
    layout.unpack_sum(state, reference, perturbation.data());

    std::vector<T> complement(block.size(), 0);
    complement[1] = -0.75;
    const auto expected = physical_block(block, complement, layout);

    fdm::NSCylSpectralFilter<T> filter(
        state.nr, state.nphi, state.nz, coordinate_projector(block));
    const auto before = layout.pack(state);
    const auto measured = filter.measure(state, reference);
    assert_true(layout.pack(state) == before);
    assert_true(std::abs(measured.removed_norm-2.0) < 1e-13);
    assert_int_equal(measured.blocks.front().coordinates_before.size(), 1);
    assert_true(std::abs(measured.blocks.front().coordinates_before[0]-2.0)
                < 1e-13);
    assert_true(std::abs(measured.blocks.front().coordinates_after[0])
                < 1e-13);
    assert_true(measured.velocity_perturbation_norm > 0);
    assert_true(measured.removed_velocity_norm > 0);
    assert_true(measured.filtered_velocity_norm > 0);

    const auto diagnostics = filter.remove(state, reference);
    std::vector<T> actual(layout.state_size);
    layout.pack_difference(state, reference, actual.data());
    assert_true(relative_error(actual, expected) < 2e-14);
    assert_int_equal(diagnostics.blocks.size(), 1);
    assert_true(std::abs(diagnostics.removed_norm-2.0) < 1e-13);
    assert_true(diagnostics.remaining_unstable_norm < 1e-13);
}

void test_filter_does_not_change_discrete_couette_state(void**) {
    const Config config = make_config();
    Task state(config);
    const Layout layout(state);
    const auto reference = couette_reference(state, layout);
    fdm::NSCylFourierBlockReference<T, true> block(config, 0, 1);
    fdm::NSCylSpectralFilter<T> filter(
        state.nr, state.nphi, state.nz, coordinate_projector(block));

    const auto diagnostics = filter.remove(state, reference);
    const auto after = layout.pack(state);
    assert_true(after == reference);
    assert_true(diagnostics.velocity_perturbation_norm == 0);
    assert_true(diagnostics.removed_velocity_norm == 0);
    assert_true(diagnostics.filtered_velocity_norm == 0);
}

void test_actual_filter_preserves_stable_mode_and_constraints(void**) {
    const Config config = make_unstable_config();
    Task state(config);
    const Layout layout(state);
    const auto reference = couette_reference(state, layout);
    fdm::NSCylFourierBlockReference<T, true> block(config, 0, 3);
    const auto spectrum = fdm::solve_ns_cyl_dense_block(
        block, state.dt, 1e-8, 1e-10);
    const auto unstable = std::max_element(
        spectrum.modes.begin(), spectrum.modes.end(),
        [](const auto& a, const auto& b) {
            return a.growth_rate < b.growth_rate;
        });
    const auto stable = std::find_if(
        spectrum.modes.begin(), spectrum.modes.end(),
        [](const auto& mode) {
            return !mode.growing && mode.residual_accepted;
        });
    assert_true(unstable != spectrum.modes.end());
    assert_true(stable != spectrum.modes.end());
    assert_true(unstable->filterable_unstable());

    fdm::NSCylSpectralModeSet<T> modes;
    modes.append_filterable_mode(*unstable);
    fdm::NSCylSpectralFilter<T> filter(
        state.nr, state.nphi, state.nz,
        fdm::NSCylSpectralProjector<T>(modes, 1e6));

    constexpr T epsilon = 1e-5;
    std::vector<T> block_state(block.size());
    std::vector<T> stable_state(block.size());
    for (int i = 0; i < block.size(); ++i) {
        stable_state[i] = epsilon*stable->right_columns[i];
        block_state[i] = stable_state[i]
            +epsilon*unstable->right_columns[i];
    }
    const auto perturbation = physical_block(block, block_state, layout);
    const auto expected = physical_block(block, stable_state, layout);
    layout.unpack_sum(state, reference, perturbation.data());

    const auto diagnostics = filter.remove(state, reference);
    std::vector<T> actual(layout.state_size);
    layout.pack_difference(state, reference, actual.data());
    const double stable_error = relative_error(actual, expected);
    const double divergence_after_filter = maximum_interior_divergence(state);
    assert_true(stable_error < 2e-11);
    assert_true(diagnostics.remaining_unstable_norm < 2e-12*epsilon);
    assert_true(std::abs(layout.pressure_mean(state)) < 2e-13);
    assert_radial_wall_conditions(state, 2e-13);
    assert_true(divergence_after_filter < 2e-11);

    state.step();
    const double divergence_after_step = maximum_interior_divergence(state);
    printf("nonlinear filter constraints: stable_error=%e "
           "div_after=%e div_next=%e\n",
           stable_error, divergence_after_filter, divergence_after_step);
    assert_finite_state(state);
    assert_radial_wall_conditions(state, 2e-13);
    assert_true(divergence_after_step < 2e-10);
}

void check_whole_block_removal(int m, int l) {
    const Config config = make_config();
    Task state(config);
    const Layout layout(state);
    const auto reference = couette_reference(state, layout);
    fdm::NSCylFourierBlockReference<T, true> block(config, m, l);
    std::vector<T> block_state(block.size());
    for (int i = 0; i < block.size(); ++i) {
        block_state[i] = std::sin(0.19*(i+1))+0.3*std::cos(0.07*(i+1));
    }
    const auto perturbation = physical_block(block, block_state, layout);
    layout.unpack_sum(state, reference, perturbation.data());

    fdm::NSCylSpectralFilter<T> filter(
        state.nr, state.nphi, state.nz, coordinate_projector(block));
    const auto diagnostics = filter.remove(
        state, reference, fdm::NSCylSpectralRemoval::whole_fourier_blocks);
    const auto actual = layout.pack(state);
    assert_true(relative_error(actual, reference) < 2e-14);
    assert_true(std::abs(diagnostics.blocks.front().removed_norm
                         -diagnostics.blocks.front().block_norm) < 1e-13);
    assert_true(diagnostics.remaining_unstable_norm < 1e-13);
}

void test_whole_block_removal_preserves_real_packing(void**) {
    check_whole_block_removal(0, 0);
    check_whole_block_removal(0, 1);
    check_whole_block_removal(1, 0);
    check_whole_block_removal(1, 1);
    check_whole_block_removal(2, 2);
}

void test_filter_normalizes_pressure_gauge(void**) {
    const Config config = make_config();
    Task state(config);
    const Layout layout(state);
    const auto reference = couette_reference(state, layout);
    for (int i = 0; i < state.p.size; ++i) {
        state.p.vec[i] += 3.0;
    }
    assert_true(std::abs(layout.pressure_mean(state)-3.0) < 1e-14);

    fdm::NSCylFourierBlockReference<T, true> block(config, 0, 1);
    fdm::NSCylSpectralFilter<T> filter(
        state.nr, state.nphi, state.nz, coordinate_projector(block));
    filter.remove(state, reference);
    assert_true(std::abs(layout.pressure_mean(state)) < 2e-14);
    const auto diagnostics = filter.measure(state, reference);
    assert_true(diagnostics.velocity_perturbation_norm == 0);
}

void test_empty_filter_does_not_change_state(void**) {
    const Config config = make_config();
    Task state(config);
    const Layout layout(state);
    const auto reference = couette_reference(state, layout);
    state.u[1][2][2] = 0.125;
    const auto before = layout.pack(state);
    const fdm::NSCylSpectralModeSet<T> no_modes;
    fdm::NSCylSpectralFilter<T> filter(
        state.nr, state.nphi, state.nz,
        fdm::NSCylSpectralProjector<T>(no_modes, 1e6));
    const auto diagnostics = filter.remove(state, reference);
    assert_true(layout.pack(state) == before);
    assert_true(diagnostics.blocks.empty());
    assert_true(diagnostics.removed_norm == 0);
}

} // namespace

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_velocity_norm_uses_cylindrical_measure),
        cmocka_unit_test(test_linear_unstable_trajectory_is_removed),
        cmocka_unit_test(test_removes_selected_mode_and_preserves_complement),
        cmocka_unit_test(test_filter_does_not_change_discrete_couette_state),
        cmocka_unit_test(
            test_actual_filter_preserves_stable_mode_and_constraints),
        cmocka_unit_test(test_whole_block_removal_preserves_real_packing),
        cmocka_unit_test(test_filter_normalizes_pressure_gauge),
        cmocka_unit_test(test_empty_filter_does_not_change_state),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
