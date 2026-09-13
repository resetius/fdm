#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <netcdf.h>

#include "config.h"
#include "ns_cyl_boundary_lqr.h"
#include "ns_cyl_boundary_lqr_storage.h"
#include "ns_cyl_state.h"

extern "C" {
#include <cmocka.h>
}

namespace {

Config make_config() {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl_boundary_lqr_storage",
        "--ns:r=1.0", "--ns:R=2.0", "--ns:h1=0.0",
        "--ns:h2=6.283185307179586", "--ns:nr=4", "--ns:nz=4",
        "--ns:nphi=4", "--ns:u0=1.0", "--ns:Re=20.0",
        "--ns:dt=0.0001", "--ns:verbose=0"
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    config.rewrite(static_cast<int>(argv.size()), argv.data());
    return config;
}

void test_gain_round_trip_preserves_global_control(void**) {
    Config config = make_config();
    const std::vector<std::pair<int, int>> blocks = {
        {0, 0}, {0, 1}, {1, 1}, {2, 2}
    };
    constexpr int horizon = 2;
    constexpr int interval = 2;
    constexpr double weight = 0.01;
    constexpr double ridge = 1e-10;
    const std::string components = "tangential";
    fdm::NSCylBoundaryLQR<double> original(
        config, blocks, horizon, interval,
        weight, ridge, components, true);

    const auto metadata =
        fdm::make_ns_cyl_boundary_lqr_gain_metadata<double>(
            config, horizon, interval, weight, ridge, components);
    const std::string filename = std::string(
        NS_CYL_BOUNDARY_LQR_TEST_DIR)+"/boundary_lqr_gain_round_trip.nc";
    const fdm::NSCylBoundaryLQRGainStorage storage(filename);
    storage.save(original.cached_gain_set(), metadata);

    fdm::NSCylBoundaryLQRGainSet<double> gains;
    fdm::NSCylBoundaryLQRGainMetadata loaded_metadata;
    storage.load(gains, loaded_metadata, metadata);
    for (const auto& gain : gains.blocks) {
        assert_false(gain.reduced_hessian.empty());
    }
    fdm::NSCylBoundaryLQR<double> restored(
        config, gains, horizon, interval,
        weight, ridge, components);
    assert_int_equal(restored.block_count(), original.block_count());

    const fdm::NSCylStateLayout<double> layout(4, 4, 4);
    std::vector<double> state(layout.state_size);
    std::vector<double> radial(16), axial(16), azimuthal(16);
    std::mt19937 generator(4519);
    std::uniform_real_distribution<double> distribution(-0.02, 0.02);
    for (double& value : state) value = distribution(generator);
    for (double& value : radial) value = distribution(generator);
    for (double& value : axial) value = distribution(generator);
    for (double& value : azimuthal) value = distribution(generator);

    const auto expected = original.control(
        state, radial, axial, azimuthal);
    const auto actual = restored.control(
        state, radial, axial, azimuthal);
    for (std::size_t index = 0; index < radial.size(); ++index) {
        assert_true(std::abs(actual.radial[index]-expected.radial[index])
                    < 1e-14);
        assert_true(std::abs(actual.axial[index]-expected.axial[index])
                    < 1e-14);
        assert_true(std::abs(
            actual.azimuthal[index]-expected.azimuthal[index]) < 1e-14);
    }

    const double limit = 0.5*expected.rms_norm();
    const auto expected_constrained = original.control(
        state, radial, axial, azimuthal, limit);
    const auto actual_constrained = restored.control(
        state, radial, axial, azimuthal, limit);
    assert_true(std::abs(actual_constrained.rms_norm()-limit) < 2e-13);
    assert_true(std::abs(actual_constrained.constraint_multiplier
                         -expected_constrained.constraint_multiplier)
                < 2e-12*std::max(
                    1.0, expected_constrained.constraint_multiplier));
    for (std::size_t index = 0; index < radial.size(); ++index) {
        assert_true(std::abs(actual_constrained.radial[index]
                             -expected_constrained.radial[index]) < 1e-14);
        assert_true(std::abs(actual_constrained.axial[index]
                             -expected_constrained.axial[index]) < 1e-14);
        assert_true(std::abs(actual_constrained.azimuthal[index]
                             -expected_constrained.azimuthal[index]) < 1e-14);
    }
    std::remove(filename.c_str());
}

void test_legacy_gain_without_reduced_hessian_remains_usable(void**) {
    Config config = make_config();
    const std::vector<std::pair<int, int>> blocks = {
        {0, 0}, {0, 1}, {1, 1}, {2, 2}
    };
    constexpr int horizon = 2;
    constexpr int interval = 2;
    constexpr double weight = 0.01;
    constexpr double ridge = 1e-10;
    const std::string components = "tangential";
    fdm::NSCylBoundaryLQR<double> original(
        config, blocks, horizon, interval,
        weight, ridge, components, true);

    auto legacy_gains = original.cached_gain_set();
    for (auto& gain : legacy_gains.blocks) {
        gain.reduced_hessian.clear();
    }
    const auto metadata =
        fdm::make_ns_cyl_boundary_lqr_gain_metadata<double>(
            config, horizon, interval, weight, ridge, components);
    const std::string filename = std::string(
        NS_CYL_BOUNDARY_LQR_TEST_DIR)+"/boundary_lqr_legacy_gain.nc";
    const fdm::NSCylBoundaryLQRGainStorage storage(filename);
    storage.save(legacy_gains, metadata);

    // Files written before constrained MPC have neither the attribute nor
    // the reduced-Hessian variables.  Reproduce that schema exactly.
    int ncid = -1;
    assert_int_equal(nc_open(filename.c_str(), NC_WRITE, &ncid), NC_NOERR);
    assert_int_equal(nc_redef(ncid), NC_NOERR);
    assert_int_equal(nc_del_att(
                         ncid, NC_GLOBAL,
                         "reduced_hessian_value_count"),
                     NC_NOERR);
    assert_int_equal(nc_enddef(ncid), NC_NOERR);
    assert_int_equal(nc_close(ncid), NC_NOERR);

    fdm::NSCylBoundaryLQRGainSet<double> loaded_gains;
    fdm::NSCylBoundaryLQRGainMetadata loaded_metadata;
    storage.load(loaded_gains, loaded_metadata, metadata);
    for (const auto& gain : loaded_gains.blocks) {
        assert_true(gain.reduced_hessian.empty());
    }
    fdm::NSCylBoundaryLQR<double> restored(
        config, loaded_gains, horizon, interval,
        weight, ridge, components);

    const fdm::NSCylStateLayout<double> layout(4, 4, 4);
    std::vector<double> state(layout.state_size);
    std::vector<double> radial(16), axial(16), azimuthal(16);
    std::mt19937 generator(4520);
    std::uniform_real_distribution<double> distribution(-0.02, 0.02);
    for (double& value : state) value = distribution(generator);
    for (double& value : radial) value = distribution(generator);
    for (double& value : axial) value = distribution(generator);
    for (double& value : azimuthal) value = distribution(generator);

    const auto expected = original.control(
        state, radial, axial, azimuthal);
    const auto actual = restored.control(
        state, radial, axial, azimuthal);
    for (std::size_t index = 0; index < radial.size(); ++index) {
        assert_float_equal(actual.radial[index], expected.radial[index], 0.0);
        assert_float_equal(actual.axial[index], expected.axial[index], 0.0);
        assert_float_equal(
            actual.azimuthal[index], expected.azimuthal[index], 0.0);
    }

    bool rejected = false;
    try {
        restored.control(
            state, radial, axial, azimuthal, 0.5*actual.rms_norm());
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("rebuild and save")
            != std::string::npos;
    }
    assert_true(rejected);
    std::remove(filename.c_str());
}

} // namespace

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_gain_round_trip_preserves_global_control),
        cmocka_unit_test(
            test_legacy_gain_without_reduced_hessian_remains_usable),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
