#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "config.h"
#include "ns_cyl_boundary_lqr.h"
#include "ns_cyl_fourier_batch.h"
#include "ns_cyl_fourier_block.h"
#include "ns_cyl_fourier_native.h"
#include "ns_cyl_spectral_modes.h"
#include "ns_cyl_spectral_projector.h"
#include "projection.h"

extern "C" {
#include <cmocka.h>
}

namespace {

Config make_config(int nr=4, int nz=4, int nphi=4) {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl_fourier_block",
        "--ns:r=1.0",
        "--ns:R=2.0",
        "--ns:h1=0.0",
        "--ns:h2=6.283185307179586",
        "--ns:nr="+std::to_string(nr),
        "--ns:nz="+std::to_string(nz),
        "--ns:nphi="+std::to_string(nphi),
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

Config make_couette_config(int nr=8, int nz=8, int nphi=8,
                           double reynolds=44.0) {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl_fourier_block",
        "--ns:r=1.5707963267948966",
        "--ns:R=3.141592653589793",
        "--ns:h1=0.0",
        "--ns:h2=10.0",
        "--ns:nr="+std::to_string(nr),
        "--ns:nz="+std::to_string(nz),
        "--ns:nphi="+std::to_string(nphi),
        "--ns:u0=1.0",
        "--ns:Re="+std::to_string(reynolds),
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

Config make_extended_couette_config() {
    Config config = make_config(8, 8, 8);
    std::vector<std::string> arguments = {
        "ut_ns_cyl_fourier_block",
        "--spectral:base_outer_radius=1.5"
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    config.rewrite(static_cast<int>(argv.size()), argv.data());
    return config;
}

void test_packed_fft2_round_trip(void**) {
    constexpr int nphi = 8;
    constexpr int nz = 4;
    fdm::PeriodicPackedFFT2<double> fft(nphi, nz);
    std::vector<double> coefficients(nphi*nz);
    std::vector<double> values(nphi*nz);
    std::vector<double> reconstructed(nphi*nz);

    std::mt19937 generator(17);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (auto& value : coefficients) {
        value = distribution(generator);
    }

    fft.synthesis(coefficients.data(), values.data());
    fft.analysis(values.data(), reconstructed.data());

    double max_error = 0;
    for (int i = 0; i < nphi*nz; ++i) {
        max_error = std::max(max_error,
                             std::abs(coefficients[i]-reconstructed[i]));
    }
    assert_true(max_error < 2e-14);
}

void check_block_round_trip(int m, int l, int expected_phases) {
    Config config = make_config();
    fdm::NSCylFourierBlockReference<double, true> block(config, m, l);
    assert_int_equal(block.phase_count(), expected_phases);
    assert_int_equal(block.full_size(), block.radial_size()*expected_phases);
    assert_int_equal(block.size(),
                     block.full_size()-((m == 0 && l == 0) ? 1 : 0));
    assert_int_equal(block.pressure_gauge_fixed(), m == 0 && l == 0);

    std::vector<double> x(block.size());
    std::vector<double> y(block.size());
    std::mt19937 generator(31+7*m+l);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (auto& value : x) {
        value = distribution(generator);
    }

    block.lift(x.data());
    block.extract(y.data());

    double max_error = 0;
    for (int i = 0; i < block.size(); ++i) {
        max_error = std::max(max_error, std::abs(x[i]-y[i]));
    }
    assert_true(max_error < 2e-14);
    assert_true(block.last_fourier_leakage() < 2e-14);
}

void test_block_layout_and_round_trip(void**) {
    check_block_round_trip(0, 0, 1);
    check_block_round_trip(0, 1, 2);
    check_block_round_trip(1, 0, 2);
    check_block_round_trip(1, 1, 4);
    check_block_round_trip(2, 2, 1);
}

void test_zero_block_uses_weighted_zero_mean_pressure(void**) {
    Config config = make_config();
    fdm::NSCylFourierBlockReference<double, true> block(config, 0, 0);
    const auto& layout = block.state_layout();
    auto& task = block.task();

    std::vector<double> full(block.full_size(), 0.0);
    std::vector<double> reduced(block.size());
    std::vector<double> reconstructed(block.full_size());

    // A constant pressure represents only a gauge change and reduces to zero.
    for (int j = 1; j <= task.nr; ++j) {
        full[layout.radial_index(
            fdm::NSCylStateLayout<double>::Component::p, j)] = 7.0;
    }
    layout.reduce_zero_gauge_block(task, full.data(), reduced.data());
    for (int index = layout.p_radial_offset; index < block.size(); ++index) {
        assert_true(reduced[index] == 0.0);
    }

    // Independent coordinates reconstruct a unique zero-mean representative.
    for (int index = 0; index < block.size(); ++index) {
        reduced[index] = 0.03*index-0.2;
    }
    layout.expand_zero_gauge_block(task, reduced.data(), reconstructed.data());
    assert_true(std::abs(layout.zero_block_pressure_mean(
        task, reconstructed.data())) < 1e-15);

    block.lift(reduced.data());
    long double weighted_sum = 0;
    long double weight = 0;
    for (int j = 1; j <= task.nr; ++j) {
        const long double r = task.r0+(j-0.5L)*task.dr;
        weighted_sum += r*task.p[0][0][j];
        weight += r;
    }
    assert_true(std::abs(static_cast<double>(weighted_sum/weight)) < 1e-15);
}

void test_linear_step_preserves_real_packed_block(void**) {
    Config config = make_config();
    fdm::NSCylFourierBlockReference<double, true> block(config, 1, 1);
    std::vector<double> x1(block.size());
    std::vector<double> x2(block.size());
    std::vector<double> sum(block.size());
    std::vector<double> y1(block.size());
    std::vector<double> y2(block.size());
    std::vector<double> ysum(block.size());

    std::mt19937 generator(91);
    std::uniform_real_distribution<double> distribution(-0.1, 0.1);
    for (int i = 0; i < block.size(); ++i) {
        x1[i] = distribution(generator);
        x2[i] = distribution(generator);
        sum[i] = x1[i]+x2[i];
    }

    block.apply(y1.data(), x1.data());
    const double leakage1 = block.last_fourier_leakage();
    block.apply(y2.data(), x2.data());
    const double leakage2 = block.last_fourier_leakage();
    block.apply(ysum.data(), sum.data());
    const double leakage_sum = block.last_fourier_leakage();

    double max_error = 0;
    double max_value = 0;
    for (int i = 0; i < block.size(); ++i) {
        max_error = std::max(max_error, std::abs(ysum[i]-y1[i]-y2[i]));
        max_value = std::max(max_value, std::abs(ysum[i]));
    }

    assert_true(max_value > 0);
    assert_true(max_error/max_value < 2e-11);
    assert_true(leakage1 < 2e-12);
    assert_true(leakage2 < 2e-12);
    assert_true(leakage_sum < 2e-12);
}

void test_fourier_block_outer_boundary_transition_is_linear(void**) {
    Config config = make_config();
    constexpr int operator_steps = 3;
    fdm::NSCylFourierBlockReference<double, true> forced_block(
        config, 1, 1, operator_steps);
    fdm::NSCylFourierBlockReference<double, true> homogeneous_block(
        config, 1, 1, operator_steps);
    fdm::NSCylFourierBlockReference<double, true> response_block(
        config, 1, 1, operator_steps);
    std::vector<double> state(forced_block.size());
    std::vector<double> zero(forced_block.size(), 0.0);
    std::vector<double> current(forced_block.outer_boundary_size());
    std::vector<double> next(forced_block.outer_boundary_size());
    std::vector<double> forced(forced_block.size());
    std::vector<double> homogeneous(forced_block.size());
    std::vector<double> response(forced_block.size());

    std::mt19937 generator(193);
    std::uniform_real_distribution<double> state_distribution(-0.1, 0.1);
    std::uniform_real_distribution<double> wall_distribution(-0.02, 0.02);
    for (double& value : state) {
        value = state_distribution(generator);
    }
    for (double& value : current) {
        value = wall_distribution(generator);
    }
    for (double& value : next) {
        value = wall_distribution(generator);
    }

    forced_block.apply_with_outer_boundary(
        forced.data(), state.data(), current.data(), next.data());
    homogeneous_block.apply(homogeneous.data(), state.data());
    response_block.apply_with_outer_boundary(
        response.data(), zero.data(), current.data(), next.data());

    double maximum_error = 0;
    double maximum_response = 0;
    for (int coordinate = 0; coordinate < forced_block.size(); ++coordinate) {
        maximum_error = std::max(maximum_error, std::abs(
            forced[coordinate]-homogeneous[coordinate]-response[coordinate]));
        maximum_response = std::max(
            maximum_response, std::abs(response[coordinate]));
    }
    assert_true(maximum_response > 0);
    assert_true(maximum_error/maximum_response < 2e-10);
    assert_true(forced_block.last_fourier_leakage() < 2e-12);
    assert_true(response_block.last_fourier_leakage() < 2e-12);

    // A subsequent homogeneous call must not inherit the last prescribed
    // boundary value from the forced transition.
    std::vector<double> homogeneous_after(forced_block.size());
    forced_block.apply(homogeneous_after.data(), state.data());
    for (int coordinate = 0; coordinate < forced_block.size(); ++coordinate) {
        assert_true(std::abs(homogeneous_after[coordinate]
                            -homogeneous[coordinate]) < 2e-12);
    }
}

template<typename T>
void check_native_outer_boundary_matches_reference(int m, int l,
                                                   int operator_steps,
                                                   double tolerance) {
    Config config = make_config(8, 8, 8);
    fdm::NSCylFourierBlockReference<T, true> reference(
        config, m, l, operator_steps);
    fdm::NSCylFourierBlockNative<T> native(
        config, m, l, operator_steps);
    assert_int_equal(native.size(), reference.size());
    assert_int_equal(native.outer_boundary_size(),
                     reference.outer_boundary_size());

    std::vector<T> state(reference.size());
    std::vector<T> current(reference.outer_boundary_size());
    std::vector<T> next(reference.outer_boundary_size());
    std::vector<T> expected(reference.size());
    std::vector<T> actual(reference.size());
    std::mt19937 generator(701+31*m+17*l+operator_steps);
    std::uniform_real_distribution<double> state_distribution(-0.02, 0.02);
    std::uniform_real_distribution<double> wall_distribution(-0.01, 0.01);
    for (T& value : state) {
        value = static_cast<T>(state_distribution(generator));
    }
    for (T& value : current) {
        value = static_cast<T>(wall_distribution(generator));
    }
    for (T& value : next) {
        value = static_cast<T>(wall_distribution(generator));
    }
    if (m == 0 && l == 0) {
        current[0] = T(0);
        next[0] = T(0);
    }

    reference.apply_with_outer_boundary(
        expected.data(), state.data(), current.data(), next.data());
    native.apply_with_outer_boundary(
        actual.data(), state.data(), current.data(), next.data());

    long double difference2 = 0;
    long double expected2 = 0;
    for (int coordinate = 0; coordinate < reference.size(); ++coordinate) {
        const long double difference =
            static_cast<long double>(actual[coordinate])-expected[coordinate];
        difference2 += difference*difference;
        expected2 += static_cast<long double>(expected[coordinate])
            *expected[coordinate];
    }
    const double relative = std::sqrt(static_cast<double>(
        difference2/std::max(expected2, std::numeric_limits<long double>::min())));
    const double leakage = reference.last_fourier_leakage();
    std::printf("native/reference %s forced block (%d,%d), steps=%d: "
                "relative=%.6e leakage=%.6e\n",
                std::is_same_v<T, double> ? "double" : "float",
                m, l, operator_steps, relative, leakage);
    assert_true(relative < tolerance);
    const double leakage_tolerance = std::is_same_v<T, double>
        ? 2e-12 : 2e-4;
    assert_true(leakage < leakage_tolerance);
}

void test_native_outer_boundary_matches_full_reference(void**) {
    for (const auto [m, l] : std::vector<std::pair<int, int>>{
             {0, 0}, {0, 1}, {1, 0}, {1, 1}, {4, 1}, {1, 4}, {4, 4}}) {
        check_native_outer_boundary_matches_reference<double>(
            m, l, 1, 2e-11);
        check_native_outer_boundary_matches_reference<double>(
            m, l, 3, 2e-10);
        check_native_outer_boundary_matches_reference<float>(
            m, l, 3, 8e-4);
    }
}

void test_physical_boundary_lqr_minimizes_one_interval_cost(void**) {
    Config config = make_config(4, 4, 4);
    constexpr int m = 1;
    constexpr int l = 1;
    constexpr int interval_steps = 2;
    constexpr double control_weight = 0.25;
    fdm::NSCylFourierBoundaryLQR<double> controller(
        config, m, l, 1, interval_steps, control_weight, 0.0);
    fdm::NSCylFourierBlockNative<double> plant(
        config, m, l, interval_steps);
    std::vector<double> state(controller.state_size());
    std::vector<double> current(controller.boundary_size());
    std::mt19937 generator(991);
    std::uniform_real_distribution<double> state_distribution(-0.02, 0.02);
    std::uniform_real_distribution<double> wall_distribution(-0.01, 0.01);
    for (double& value : state) {
        value = state_distribution(generator);
    }
    for (double& value : current) {
        value = wall_distribution(generator);
    }

    fdm::NSCylBoundaryLQRBlockDiagnostics<double> diagnostics;
    const auto optimum = controller.control(
        state.data(), current.data(), &diagnostics);
    std::vector<double> image(controller.state_size());
    auto objective = [&](const std::vector<double>& boundary) {
        plant.apply_with_outer_boundary(
            image.data(), state.data(), current.data(), boundary.data());
        long double effort = 0;
        for (int component = 0; component < 3; ++component) {
            for (int i = 0; i < plant.nphi; ++i) {
                for (int k = 0; k < plant.nz; ++k) {
                    long double value = 0;
                    for (int phase = 0; phase < plant.phase_count(); ++phase) {
                        value += boundary[
                            component*plant.phase_count()+phase]
                            *plant.phase_value(phase, i, k);
                    }
                    effort += value*value;
                }
            }
        }
        effort /= plant.nphi*plant.nz;
        return controller.velocity_inner_product(
            image.data(), image.data())+control_weight*effort;
    };

    std::vector<double> zero(controller.boundary_size(), 0.0);
    const double zero_cost = objective(zero);
    const double optimum_cost = objective(optimum);
    assert_true(optimum_cost < zero_cost);
    assert_true(std::abs(zero_cost-diagnostics.predicted_cost_before)
                < 2e-12*std::max(1.0, zero_cost));
    assert_true(std::abs(optimum_cost-diagnostics.predicted_cost_after)
                < 2e-12*std::max(1.0, optimum_cost));

    const double epsilon = 1e-5;
    for (int coordinate = 0; coordinate < controller.boundary_size();
         ++coordinate) {
        auto plus = optimum;
        auto minus = optimum;
        plus[coordinate] += epsilon;
        minus[coordinate] -= epsilon;
        assert_true(objective(plus) >= optimum_cost-2e-13);
        assert_true(objective(minus) >= optimum_cost-2e-13);
    }
}

void test_physical_boundary_lqr_global_packing_matches_block(void**) {
    Config config = make_config(4, 4, 4);
    constexpr int m = 1;
    constexpr int l = 1;
    fdm::NSCylFourierBoundaryLQR<double> block_controller(
        config, m, l, 1, 1, 0.1, 0.0);
    fdm::NSCylBoundaryLQR<double> global_controller(
        config, {{m, l}}, 1, 1, 0.1, 0.0);
    fdm::NSCylFourierBlockReference<double, true> reference(
        config, m, l, 1);
    const fdm::NSCylStateLayout<double> layout(reference.task());
    std::vector<double> block_state(reference.size());
    std::vector<double> current_coefficients(reference.outer_boundary_size());
    std::mt19937 generator(1223);
    std::uniform_real_distribution<double> distribution(-0.02, 0.02);
    for (double& value : block_state) {
        value = distribution(generator);
    }
    for (double& value : current_coefficients) {
        value = distribution(generator);
    }
    reference.lift(block_state.data());
    const auto physical_state = layout.pack(reference.task());

    fdm::PeriodicPackedFFT2<double> fft(4, 4);
    std::vector<double> plane_coefficients(16, 0.0);
    std::vector<std::vector<double>> current(
        3, std::vector<double>(16, 0.0));
    int component = 0;
    for (int component_index = 0; component_index < 3; ++component_index) {
        std::fill(plane_coefficients.begin(), plane_coefficients.end(), 0.0);
        int phase = 0;
        for (int i : reference.phi_indices()) {
            for (int k : reference.z_indices()) {
                plane_coefficients[i*4+k] = current_coefficients[
                    component_index*reference.phase_count()+phase];
                ++phase;
            }
        }
        fft.synthesis(plane_coefficients.data(), current[component].data());
        ++component;
    }

    const auto expected = block_controller.control(
        block_state.data(), current_coefficients.data());
    const auto actual = global_controller.control(
        physical_state, current[0], current[1], current[2]);
    std::vector<std::vector<double>> actual_physical = {
        actual.radial, actual.axial, actual.azimuthal
    };
    for (int component_index = 0; component_index < 3; ++component_index) {
        fft.analysis(
            actual_physical[component_index].data(),
            plane_coefficients.data());
        int phase = 0;
        for (int i : reference.phi_indices()) {
            for (int k : reference.z_indices()) {
                assert_true(std::abs(
                    plane_coefficients[i*4+k]
                    -expected[component_index*reference.phase_count()+phase])
                    < 2e-13);
                ++phase;
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int k = 0; k < 4; ++k) {
                const bool selected = std::find(
                    reference.phi_indices().begin(),
                    reference.phi_indices().end(), i)
                        != reference.phi_indices().end()
                    && std::find(
                        reference.z_indices().begin(),
                        reference.z_indices().end(), k)
                        != reference.z_indices().end();
                if (!selected) {
                    assert_true(std::abs(plane_coefficients[i*4+k]) < 2e-13);
                }
            }
        }
    }
}

void test_physical_boundary_lqr_respects_component_selection(void**) {
    Config config = make_config(4, 4, 4);
    fdm::NSCylFourierBoundaryLQR<double> tangential(
        config, 1, 1, 1, 1, 0.1, 0.0, "tangential");
    fdm::NSCylFourierBoundaryLQR<double> azimuthal(
        config, 1, 1, 1, 1, 0.1, 0.0, "azimuthal");
    assert_int_equal(tangential.input_size(), 2*tangential.phase_count());
    assert_int_equal(azimuthal.input_size(), azimuthal.phase_count());
    std::vector<double> state(tangential.state_size(), 0.0);
    std::vector<double> current(tangential.boundary_size(), 0.0);
    state[0] = 0.01;
    const auto tangential_control = tangential.control(
        state.data(), current.data());
    const auto azimuthal_control = azimuthal.control(
        state.data(), current.data());
    for (int phase = 0; phase < tangential.phase_count(); ++phase) {
        assert_float_equal(tangential_control[phase], 0.0, 0.0);
        assert_float_equal(azimuthal_control[phase], 0.0, 0.0);
        assert_float_equal(
            azimuthal_control[azimuthal.phase_count()+phase], 0.0, 0.0);
    }
}

void test_batched_blocks_match_individual_applications(void**) {
    Config config = make_config();
    constexpr int operator_steps = 3;
    const std::vector<std::pair<int, int>> indices = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 2}
    };
    const double scales[] = {1e-6, 1e-2, 1.0, 1e2, 1e6};

    fdm::NSCylFourierBlockBatchReference<double, true> batch(
        config, operator_steps);
    std::vector<std::vector<double>> input(indices.size());
    std::vector<std::vector<double>> batched(indices.size());
    std::vector<std::vector<double>> individual(indices.size());
    std::vector<fdm::NSCylFourierBatchRequest<double>> requests;
    std::mt19937 generator(117);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (std::size_t block_index = 0;
         block_index < indices.size(); ++block_index) {
        const auto [m, l] = indices[block_index];
        fdm::NSCylFourierBlockReference<double, true> block(
            config, m, l, operator_steps);
        input[block_index].resize(block.size());
        batched[block_index].resize(block.size());
        individual[block_index].resize(block.size());
        for (double& value : input[block_index]) {
            value = scales[block_index]*distribution(generator);
        }
        block.apply(individual[block_index].data(), input[block_index].data());
        requests.push_back({
            m, l, input[block_index].data(), batched[block_index].data(),
            block.size()});
    }

    batch.apply(requests);

    for (std::size_t block_index = 0;
         block_index < indices.size(); ++block_index) {
        double max_error = 0;
        double max_value = 0;
        for (std::size_t i = 0; i < batched[block_index].size(); ++i) {
            max_error = std::max(max_error, std::abs(
                batched[block_index][i]-individual[block_index][i]));
            max_value = std::max(
                max_value, std::abs(individual[block_index][i]));
        }
        assert_true(max_value > 0);
        assert_true(max_error/max_value < 3e-11);
    }
}

template<typename T>
void check_native_radial_blocks(int operator_steps, double tolerance) {
    Config config = make_config(8, 8, 8);
    const std::vector<std::pair<int, int>> indices = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1},
        {4, 1}, {1, 4}, {4, 4}
    };
    std::mt19937 generator(991+operator_steps);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (const auto [m, l] : indices) {
        fdm::NSCylFourierBlockReference<T, true> reference(
            config, m, l, operator_steps);
        fdm::NSCylFourierBlockNative<T> native(
            config, m, l, operator_steps);
        assert_int_equal(native.size(), reference.size());
        assert_int_equal(native.phase_count(), reference.phase_count());

        std::vector<T> input(reference.size());
        std::vector<T> expected(reference.size());
        std::vector<T> actual(reference.size());
        for (T& value : input) {
            value = static_cast<T>(distribution(generator));
        }
        reference.apply(expected.data(), input.data());
        native.apply(actual.data(), input.data());

        long double error2 = 0;
        long double expected2 = 0;
        long double velocity_error2 = 0;
        long double velocity_expected2 = 0;
        long double pressure_error2 = 0;
        long double pressure_expected2 = 0;
        double maximum_error = 0;
        const int pressure_offset = reference.state_layout().p_radial_offset;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const long double error =
                static_cast<long double>(actual[i])-expected[i];
            error2 += error*error;
            expected2 += static_cast<long double>(expected[i])*expected[i];
            const bool pressure =
                static_cast<int>(i%reference.radial_size()) >= pressure_offset;
            if (pressure) {
                pressure_error2 += error*error;
                pressure_expected2 +=
                    static_cast<long double>(expected[i])*expected[i];
            } else {
                velocity_error2 += error*error;
                velocity_expected2 +=
                    static_cast<long double>(expected[i])*expected[i];
            }
            maximum_error = std::max(
                maximum_error, static_cast<double>(std::abs(error)));
        }
        const double relative = static_cast<double>(
            std::sqrt(error2/expected2));
        const double velocity_relative = static_cast<double>(
            std::sqrt(velocity_error2/velocity_expected2));
        const double pressure_relative = static_cast<double>(
            std::sqrt(pressure_error2/pressure_expected2));
        printf("native/reference %s block (%d,%d), steps=%d: "
               "relative=%e velocity=%e pressure=%e max=%e\n",
               std::is_same_v<T, float> ? "float" : "double",
               m, l, operator_steps, relative, velocity_relative,
               pressure_relative, maximum_error);
        assert_true(relative < tolerance);
        if constexpr (std::is_same_v<T, float>) {
            // The pressure itself is more sensitive to operation order in the
            // nearly-Neumann solve.  The projected velocity, which is the
            // dynamical state used by the filter, must still agree tightly.
            assert_true(velocity_relative < 2e-6);
            assert_true(pressure_relative < 6e-4);
        }
    }
}

void test_native_radial_blocks_match_full_reference(void**) {
    check_native_radial_blocks<double>(1, 2e-11);
    check_native_radial_blocks<double>(3, 5e-11);
    check_native_radial_blocks<float>(3, 3e-4);
}

void test_zero_extended_couette_base_matches_native_operator(void**) {
    Config config = make_extended_couette_config();
    constexpr int operator_steps = 3;
    fdm::NSCylFourierBlockReference<double, true> reference(
        config, 1, 1, operator_steps);
    fdm::NSCylFourierBlockNative<double> native(
        config, 1, 1, operator_steps);

    auto& state = reference.task();
    assert_true(std::abs(state.w0[0][0][4]) > 1e-3);
    for (int j = 5; j <= state.nr+1; ++j) {
        assert_true(state.w0[0][0][j] == 0);
    }

    std::vector<double> input(reference.size());
    std::vector<double> expected(reference.size());
    std::vector<double> actual(reference.size());
    for (int i = 0; i < reference.size(); ++i) {
        input[i] = std::sin(0.071*(i+1))+0.2*std::cos(0.113*(i+1));
    }
    reference.apply(expected.data(), input.data());
    native.apply(actual.data(), input.data());

    long double error2 = 0;
    long double expected2 = 0;
    for (int i = 0; i < reference.size(); ++i) {
        const long double error = actual[i]-expected[i];
        error2 += error*error;
        expected2 += static_cast<long double>(expected[i])*expected[i];
    }
    assert_true(std::sqrt(error2/expected2) < 5e-11);

    Config misaligned = make_config(8, 8, 8);
    std::vector<std::string> arguments = {
        "ut_ns_cyl_fourier_block",
        "--spectral:base_outer_radius=1.51"
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    misaligned.rewrite(static_cast<int>(argv.size()), argv.data());
    bool rejected = false;
    try {
        fdm::NSCylFourierBlockNative<double> invalid(misaligned, 0, 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert_true(rejected);
}

void test_zero_extended_base_matches_nonlinear_central_difference(void**) {
    using Task = fdm::NSCyl<double, true, fdm::tensor_flag::periodic>;
    constexpr double epsilon = 1e-5;
    Config config = make_extended_couette_config();
    Task plus(config);
    Task minus(config);
    Task linear(config);
    const fdm::NSCylStateLayout<double> layout(linear);

    layout.initialize_couette_state(plus, 1.5);
    const auto reference = layout.pack(plus);
    std::vector<double> perturbation(layout.state_size);
    std::vector<double> positive(layout.state_size);
    std::vector<double> negative(layout.state_size);
    for (int index = 0; index < layout.state_size; ++index) {
        perturbation[index] =
            0.03*std::sin(0.071*(index+1))
            +0.01*std::cos(0.113*(index+1));
        positive[index] = reference[index]+epsilon*perturbation[index];
        negative[index] = reference[index]-epsilon*perturbation[index];
    }
    layout.unpack(plus, positive.data());
    layout.unpack(minus, negative.data());
    layout.unpack(linear, perturbation.data());
    layout.initialize_couette_linearization(linear, 1.5);
    linear.U0 = 0;

    plus.step();
    minus.step();
    linear.L_step();
    const auto positive_image = layout.pack(plus);
    const auto negative_image = layout.pack(minus);
    const auto linear_image = layout.pack(linear);
    double maximum_error = 0;
    double maximum_reference = 0;
    for (int index = 0; index < layout.state_size; ++index) {
        const double derivative =
            (positive_image[index]-negative_image[index])/(2*epsilon);
        maximum_error = std::max(
            maximum_error, std::abs(linear_image[index]-derivative));
        maximum_reference = std::max(
            maximum_reference, std::abs(derivative));
    }
    assert_true(maximum_reference > 0);
    assert_true(maximum_error/maximum_reference < 2e-8);
}

void test_axisymmetric_block_is_independent_of_nphi(void**) {
    Config coarse = make_couette_config(8, 16, 8, 100.0);
    Config refined = make_couette_config(8, 16, 16, 100.0);
    fdm::NSCylFourierBlockReference<double, true> coarse_block(
        coarse, 0, 2, 3);
    fdm::NSCylFourierBlockReference<double, true> refined_block(
        refined, 0, 2, 3);

    assert_int_equal(coarse_block.size(), refined_block.size());
    std::vector<double> x(coarse_block.size());
    std::vector<double> coarse_image(x.size());
    std::vector<double> refined_image(x.size());
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        x[i] = std::sin(0.17*(i+1))+0.2*std::cos(0.31*(i+1));
    }

    coarse_block.apply(coarse_image.data(), x.data());
    refined_block.apply(refined_image.data(), x.data());

    double max_error = 0;
    double max_value = 0;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        max_error = std::max(
            max_error, std::abs(coarse_image[i]-refined_image[i]));
        max_value = std::max(max_value, std::abs(refined_image[i]));
    }
    assert_true(max_value > 0);
    assert_true(max_error/max_value < 2e-12);
}

void test_dense_spectrum_groups_complex_pair_in_real_columns(void**) {
    // The 2x2 block has eigenvalues 1.1 +/- 0.2i. Coupling from the third
    // coordinate makes the full real matrix nonsymmetric and nonnormal.
    const double matrix[] = {
        1.1,  0.2, 0.0,
       -0.2,  1.1, 0.0,
        0.4, -0.1, 0.8
    };
    const double duration = 0.5;
    auto spectrum = fdm::analyze_ns_cyl_dense_matrix(
        matrix, 3, duration, 0.0, 1e-12);

    assert_int_equal(spectrum.block_size, 3);
    assert_int_equal(spectrum.modes.size(), 2);
    assert_true(spectrum.max_right_residual < 1e-14);
    assert_true(spectrum.max_left_residual < 1e-14);

    const fdm::NSCylSpectralMode<double>* pair = nullptr;
    const fdm::NSCylSpectralMode<double>* real_mode = nullptr;
    for (const auto& mode : spectrum.modes) {
        if (mode.column_count == 2) {
            pair = &mode;
        } else {
            real_mode = &mode;
        }
    }
    assert_non_null(pair);
    assert_non_null(real_mode);
    assert_int_equal(pair->right_columns.size(), 6);
    assert_int_equal(pair->left_columns.size(), 6);
    assert_true(std::abs(pair->multiplier.real()-1.1) < 1e-14);
    assert_true(std::abs(pair->multiplier.imag()-0.2) < 1e-14);
    assert_true(std::abs(pair->growth_rate
        -std::log(std::hypot(1.1, 0.2))/duration) < 1e-14);
    assert_true(std::abs(pair->frequency
        -std::atan2(0.2, 1.1)/duration) < 1e-14);
    assert_true(pair->filterable_unstable());
    assert_false(real_mode->growing);

    fdm::NSCylSpectralModeSet<double> unstable_modes;
    unstable_modes.append_filterable(spectrum);
    unstable_modes.sort_by_block_and_growth();
    const fdm::NSCylSpectralProjector<double> projector(
        unstable_modes, 1e8);
    assert_int_equal(projector.blocks().size(), 1);
    const auto& block_projector = projector.blocks().front();
    assert_int_equal(block_projector.dimension(), 2);
    assert_true(block_projector.condition_number() >= 1.0);
    assert_true(block_projector.condition_number() < 10.0);

    std::vector<double> projected(3);
    for (const auto& vector : block_projector.right_basis()) {
        block_projector.project(projected.data(), vector.data());
        for (int i = 0; i < 3; ++i) {
            assert_true(std::abs(projected[i]-vector[i]) < 1e-14);
        }
    }
    block_projector.project(projected.data(), real_mode->right_columns.data());
    for (double value : projected) {
        assert_true(std::abs(value) < 1e-14);
    }

    const std::vector<double> state = {0.3, -0.7, 1.1};
    std::vector<double> filtered(3);
    block_projector.remove(filtered.data(), state.data());
    block_projector.project(projected.data(), filtered.data());
    for (double value : projected) {
        assert_true(std::abs(value) < 1e-14);
    }

    for (int i = 0; i+1 < 3; ++i) {
        if (spectrum.eigenvalues[i].imag() > 0) {
            assert_true(std::abs(spectrum.eigenvalues[i].real()
                -spectrum.eigenvalues[i+1].real()) < 1e-14);
            assert_true(std::abs(spectrum.eigenvalues[i].imag()
                +spectrum.eigenvalues[i+1].imag()) < 1e-14);
        }
    }

    auto later_block = spectrum;
    auto earlier_block = spectrum;
    auto faster_earlier_block = spectrum;
    for (auto& mode : later_block.modes) {
        mode.m = 2;
        mode.l = 1;
    }
    for (auto& mode : earlier_block.modes) {
        mode.m = 0;
        mode.l = 3;
    }
    for (auto& mode : faster_earlier_block.modes) {
        mode.m = 0;
        mode.l = 3;
        mode.growth_rate += 1.0;
    }
    fdm::NSCylSpectralModeSet<double> modes;
    modes.append_filterable(later_block);
    modes.append_filterable(earlier_block);
    modes.append_filterable(faster_earlier_block);
    modes.sort_by_block_and_growth();
    assert_int_equal(modes.size(), 3);
    assert_int_equal(modes.real_dimension(), 6);
    assert_int_equal(modes.modes()[0].m, 0);
    assert_int_equal(modes.modes()[0].l, 3);
    assert_int_equal(modes.modes()[1].m, 0);
    assert_int_equal(modes.modes()[1].l, 3);
    assert_true(modes.modes()[0].growth_rate
                > modes.modes()[1].growth_rate);
    assert_int_equal(modes.modes()[2].m, 2);
    assert_int_equal(modes.modes()[2].l, 1);

    // Orthonormalization removes arbitrary vector scaling, while nearly
    // orthogonal left and right subspaces still fail condition_limit.
    auto ill_conditioned_mode = *pair;
    ill_conditioned_mode.m = 4;
    ill_conditioned_mode.l = 2;
    ill_conditioned_mode.phase_count = 1;
    ill_conditioned_mode.radial_size = 3;
    ill_conditioned_mode.block_size = 3;
    ill_conditioned_mode.right_columns = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0
    };
    ill_conditioned_mode.left_columns = {
        1.0, 0.0, 0.0,
        0.0, 1e-12, 1.0
    };
    fdm::NSCylDenseBlockSpectrum<double> ill_conditioned_spectrum;
    ill_conditioned_spectrum.modes.push_back(ill_conditioned_mode);
    fdm::NSCylSpectralModeSet<double> ill_conditioned_modes;
    ill_conditioned_modes.append_filterable(ill_conditioned_spectrum);
    bool rejected = false;
    try {
        const fdm::NSCylSpectralProjector<double> rejected_projector(
            ill_conditioned_modes, 1e8);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert_true(rejected);
}

void test_dense_spectrum_negative_threshold_selects_slow_stable_modes(void**) {
    const double duration = 0.5;
    const double matrix[] = {
        std::exp(0.10*duration), 0.0, 0.0,
        0.0, std::exp(-0.015*duration), 0.0,
        0.0, 0.0, std::exp(-0.030*duration)
    };
    auto spectrum = fdm::analyze_ns_cyl_dense_matrix(
        matrix, 3, duration, -0.02, 1e-12);

    int selected = 0;
    bool found_unstable = false;
    bool found_slow_stable = false;
    bool found_fast_stable = false;
    for (const auto& mode : spectrum.modes) {
        if (mode.filterable_unstable()) {
            ++selected;
        }
        if (std::abs(mode.growth_rate-0.10) < 1e-13) {
            found_unstable = mode.filterable_unstable();
        } else if (std::abs(mode.growth_rate+0.015) < 1e-13) {
            found_slow_stable = mode.filterable_unstable();
        } else if (std::abs(mode.growth_rate+0.030) < 1e-13) {
            found_fast_stable = !mode.growing;
        }
    }

    assert_int_equal(selected, 2);
    assert_true(found_unstable);
    assert_true(found_slow_stable);
    assert_true(found_fast_stable);
}

void test_dense_spectrum_of_real_ns_cyl_block(void**) {
    Config config = make_couette_config();
    fdm::NSCylFourierBlockReference<double, true> block(config, 0, 3);
    const auto spectrum = fdm::solve_ns_cyl_dense_block(
        block, 0.001, 1e-8, 1e-10);

    assert_int_equal(spectrum.m, 0);
    assert_int_equal(spectrum.l, 3);
    assert_int_equal(spectrum.phase_count, 2);
    assert_int_equal(spectrum.radial_size, 4*8-1);
    assert_int_equal(spectrum.block_size, 2*spectrum.radial_size);
    assert_int_equal(spectrum.operator_steps, 1);
    assert_int_equal(spectrum.operator_calls, spectrum.block_size);
    assert_false(spectrum.pressure_gauge_fixed);
    assert_true(spectrum.max_fourier_leakage < 2e-12);
    assert_true(spectrum.max_right_residual < 1e-11);
    assert_true(spectrum.max_left_residual < 1e-11);

    int real_columns = 0;
    for (const auto& mode : spectrum.modes) {
        assert_int_equal(mode.m, 0);
        assert_int_equal(mode.l, 3);
        assert_true(mode.column_count == 1 || mode.column_count == 2);
        assert_int_equal(mode.right_columns.size(),
                         mode.column_count*spectrum.block_size);
        assert_int_equal(mode.left_columns.size(),
                         mode.column_count*spectrum.block_size);
        real_columns += mode.column_count;
    }
    assert_int_equal(real_columns, spectrum.block_size);

    auto selected_spectrum = spectrum;
    for (auto& mode : selected_spectrum.modes) {
        mode.growing = false;
    }
    std::vector<int> indices(selected_spectrum.modes.size());
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return std::abs(selected_spectrum.modes[a].multiplier)
            > std::abs(selected_spectrum.modes[b].multiplier);
    });
    int selected_columns = 0;
    for (int index : indices) {
        selected_spectrum.modes[index].growing = true;
        selected_columns += selected_spectrum.modes[index].column_count;
        if (selected_columns >= 2) {
            break;
        }
    }

    fdm::NSCylSpectralModeSet<double> selected_modes;
    selected_modes.append_filterable(selected_spectrum);
    selected_modes.sort_by_block_and_growth();
    const fdm::NSCylSpectralProjector<double> projector(selected_modes, 1e8);
    const auto* block_projector = projector.find_block(0, 3);
    assert_non_null(block_projector);
    assert_int_equal(block_projector->dimension(), selected_columns);
    assert_true(block_projector->condition_number() < 1e8);

    std::vector<double> state(spectrum.block_size);
    for (int i = 0; i < spectrum.block_size; ++i) {
        state[i] = std::sin(0.17*(i+1))+0.2*std::cos(0.31*(i+1));
    }
    std::vector<double> filtered(spectrum.block_size);
    std::vector<double> remaining_projection(spectrum.block_size);
    block_projector->remove(filtered.data(), state.data());
    block_projector->project(
        remaining_projection.data(), filtered.data());
    double projection_norm = 0;
    double state_norm = 0;
    for (int i = 0; i < spectrum.block_size; ++i) {
        projection_norm += remaining_projection[i]*remaining_projection[i];
        state_norm += state[i]*state[i];
    }
    assert_true(std::sqrt(projection_norm/state_norm) < 1e-12);
}

void test_complex_ns_cyl_mode_has_expected_phase_speed(void**) {
    Config config = make_couette_config(8, 8, 8, 100.0);
    fdm::NSCylFourierBlockReference<double, true> block(config, 1, 3);
    const auto spectrum = fdm::solve_ns_cyl_dense_block(
        block, 0.001, 1e-8, 1e-10);
    const auto mode = std::max_element(
        spectrum.modes.begin(), spectrum.modes.end(),
        [](const auto& first, const auto& second) {
            return first.growth_rate < second.growth_rate;
        });
    assert_true(mode != spectrum.modes.end());
    assert_int_equal(mode->column_count, 2);
    assert_true(mode->filterable_unstable());

    const int n = block.size();
    const double* real = mode->right_columns.data();
    const double* imaginary = real+n;
    std::vector<double> state(real, real+n);
    std::vector<double> image(n);
    std::vector<double> expected(n);
    std::complex<double> multiplier_power(1.0, 0.0);
    constexpr int steps = 32;
    double maximum_relative_error = 0;
    for (int step = 0; step < steps; ++step) {
        block.apply(image.data(), state.data());
        multiplier_power *= mode->multiplier;
        long double error_squared = 0;
        long double expected_squared = 0;
        for (int row = 0; row < n; ++row) {
            expected[row] = multiplier_power.real()*real[row]
                -multiplier_power.imag()*imaginary[row];
            const long double error = image[row]-expected[row];
            error_squared += error*error;
            expected_squared += static_cast<long double>(expected[row])
                *expected[row];
        }
        maximum_relative_error = std::max(
            maximum_relative_error,
            std::sqrt(static_cast<double>(error_squared/expected_squared)));
        state.swap(image);
    }

    long double rr = 0;
    long double ri = 0;
    long double ii = 0;
    long double rx = 0;
    long double ix = 0;
    for (int row = 0; row < n; ++row) {
        rr += real[row]*real[row];
        ri += real[row]*imaginary[row];
        ii += imaginary[row]*imaginary[row];
        rx += real[row]*state[row];
        ix += imaginary[row]*state[row];
    }
    const long double determinant = rr*ii-ri*ri;
    const double real_coefficient = static_cast<double>((ii*rx-ri*ix)
                                                         /determinant);
    const double imaginary_coefficient = static_cast<double>((rr*ix-ri*rx)
                                                              /determinant);
    const double measured_frequency = std::atan2(
        -imaginary_coefficient, real_coefficient)/(steps*0.001);
    printf("complex mode phase: expected=%+.9e measured=%+.9e "
           "trajectory_error=%.3e\n",
           mode->frequency, measured_frequency, maximum_relative_error);
    assert_true(maximum_relative_error < 2e-11);
    assert_true(std::abs(measured_frequency-mode->frequency) < 2e-10);
}


// Спектральный проектор на настоящем операторе блока, а не на модельной
// матрице. По содержанию это chafe2d_check_projection2 / bar_check_projection2
// из main-2008.1: там Pp/Pm тоже применялись к состоянию реальной модели, но
// невязка печаталась, а не проверялась.
//
// operator_steps разводит спектр: за один шаг все mu сидят вплотную к единице,
// собственные вектора плохо обусловлены и матрица Грама почти вырождена.
void test_spectral_projector_on_block(void**) {
    Config config = make_config();
    fdm::NSCylFourierBlockReference<double, true> block(config, 0, 1, 100);
    const int n = block.size();
    const auto spectrum = fdm::solve_ns_cyl_dense_block(
        block, 0.0001, -INFINITY, 1e-10);

    std::vector<int> groups(spectrum.modes.size());
    for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
        groups[i] = i;
    }
    std::sort(groups.begin(), groups.end(), [&](int x, int y) {
        return std::abs(spectrum.modes[x].multiplier)
            > std::abs(spectrum.modes[y].multiplier);
    });
    // Use the leading groups as a stand-in for an unstable subspace.
    auto column_of = [&](const fdm::NSCylSpectralMode<double>& mode,
                         bool left, int column) {
        const auto& values = left ? mode.left_columns : mode.right_columns;
        return std::vector<double>(
            values.begin()+static_cast<std::size_t>(column)*n,
            values.begin()+static_cast<std::size_t>(column+1)*n);
    };

    std::vector<std::vector<double>> e, et;
    std::size_t used = 0;
    std::complex<double> cluster_edge;
    while (used < groups.size()) {
        const auto& next = spectrum.modes[groups[used]];
        if (e.size() >= 4
            && std::abs(next.multiplier-cluster_edge) > 1e-10) {
            break;
        }
        const auto& mode = spectrum.modes[groups[used++]];
        for (int k = 0; k < mode.column_count; ++k) {
            e.push_back(column_of(mode, false, k));
            et.push_back(column_of(mode, true, k));
        }
        cluster_edge = mode.multiplier;
    }
    const int m = static_cast<int>(e.size());
    assert_true(m >= 2);
    assert_true(used < groups.size());

    // The selected left and right bases have a nonsingular Gram matrix.
    std::vector<double> ete(static_cast<std::size_t>(m)*m);
    const double pivot = fdm::inverse_gramm_matrix(ete.data(), e, et, m, n);
    assert_true(pivot > 1e-8);

    auto project = [&](const std::vector<double>& h) {
        std::vector<double> result(n);
        fdm::projection2(result.data(), h.data(), e, et, ete.data(), m, n);
        return result;
    };

    // GEEV normalizes eigenvectors, so an absolute tolerance is appropriate.
    const double tol = 1e-12;

    // P r_j = r_j.
    for (int i = 0; i < m; ++i) {
        const auto projected = project(e[i]);
        for (int k = 0; k < n; ++k) {
            assert_true(std::abs(projected[k]-e[i][k]) < tol);
        }
    }

    // A vector from a different invariant subspace satisfies P r = 0.
    {
        const auto outside = column_of(
            spectrum.modes[groups[used]], false, 0);
        const auto projected = project(outside);
        for (int k = 0; k < n; ++k) {
            assert_true(std::abs(projected[k]) < tol);
        }
    }

    std::mt19937 generator(17);
    std::uniform_real_distribution<double> distribution(-1, 1);
    std::vector<double> h(n);
    for (int k = 0; k < n; ++k) {
        h[k] = distribution(generator);
    }

    // P*P = P and P+ + P- = I.
    const auto Ph = project(h);
    const auto PPh = project(Ph);
    for (int k = 0; k < n; ++k) {
        assert_true(std::abs(PPh[k]-Ph[k]) < tol);
    }

    std::vector<double> Mh(n);
    for (int k = 0; k < n; ++k) {
        Mh[k] = h[k]-Ph[k];
    }
    const auto MMh_source = project(Mh);
    for (int k = 0; k < n; ++k) {
        assert_true(std::abs(MMh_source[k]) < tol);
        assert_true(std::abs(Ph[k]+Mh[k]-h[k]) < tol);
    }

    // главное: подпространство инвариантно, значит проектор коммутирует с
    // оператором -- A P h = P A h. Слева применяем сам L_step, а не матрицу.
    std::vector<double> a_of_Ph(n);
    std::vector<double> a_of_h(n);
    block.apply(a_of_Ph.data(), Ph.data());
    block.apply(a_of_h.data(), h.data());
    const auto P_of_ah = project(a_of_h);

    double max_error = 0;
    double max_value = 0;
    for (int k = 0; k < n; ++k) {
        max_error = std::max(max_error, std::abs(a_of_Ph[k]-P_of_ah[k]));
        max_value = std::max(max_value, std::abs(a_of_Ph[k]));
    }
    assert_true(max_value > 0);
    assert_true(max_error/max_value < 1e-10);
}

void test_float_block_apply_is_finite_and_nonzero(void**) {
    Config config = make_config();
    fdm::NSCylFourierBlockReference<float, true> block(config, 0, 1);
    std::vector<float> x(block.size());
    std::vector<float> y(block.size());
    for (int i = 0; i < block.size(); ++i) {
        const float index = static_cast<float>(i+1);
        x[i] = std::sin(0.371f*index)+0.5f*std::cos(0.193f*index+0.11f);
    }

    block.apply(y.data(), x.data());

    double norm2 = 0;
    for (float value : y) {
        assert_true(std::isfinite(value));
        norm2 += static_cast<double>(value)*value;
    }
    assert_true(norm2 > 0);
    assert_true(block.last_fourier_leakage() < 2e-5);
}

} // namespace

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_packed_fft2_round_trip),
        cmocka_unit_test(test_block_layout_and_round_trip),
        cmocka_unit_test(test_zero_block_uses_weighted_zero_mean_pressure),
        cmocka_unit_test(test_linear_step_preserves_real_packed_block),
        cmocka_unit_test(
            test_fourier_block_outer_boundary_transition_is_linear),
        cmocka_unit_test(
            test_native_outer_boundary_matches_full_reference),
        cmocka_unit_test(
            test_physical_boundary_lqr_minimizes_one_interval_cost),
        cmocka_unit_test(
            test_physical_boundary_lqr_global_packing_matches_block),
        cmocka_unit_test(
            test_physical_boundary_lqr_respects_component_selection),
        cmocka_unit_test(test_batched_blocks_match_individual_applications),
        cmocka_unit_test(test_native_radial_blocks_match_full_reference),
        cmocka_unit_test(test_zero_extended_couette_base_matches_native_operator),
        cmocka_unit_test(test_zero_extended_base_matches_nonlinear_central_difference),
        cmocka_unit_test(test_axisymmetric_block_is_independent_of_nphi),
        cmocka_unit_test(test_dense_spectrum_groups_complex_pair_in_real_columns),
        cmocka_unit_test(
            test_dense_spectrum_negative_threshold_selects_slow_stable_modes),
        cmocka_unit_test(test_dense_spectrum_of_real_ns_cyl_block),
        cmocka_unit_test(test_complex_ns_cyl_mode_has_expected_phase_speed),
        cmocka_unit_test(test_spectral_projector_on_block),
        cmocka_unit_test(test_float_block_apply_is_finite_and_nonzero),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
