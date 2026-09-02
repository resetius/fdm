#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <netcdf.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "config.h"
#include "ns_cyl.h"
#include "ns_cyl_checkpoint_storage.h"
#include "ns_cyl_state.h"

extern "C" {
#include <cmocka.h>
}

namespace {

std::string test_file(const char* name) {
    return std::string(NS_CYL_CHECKPOINT_TEST_DIR)+"/"+name;
}

Config make_config(double reynolds=44.0) {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl_checkpoint_storage",
        "--ns:r=1.0",
        "--ns:R=2.0",
        "--ns:h1=0.0",
        "--ns:h2=6.0",
        "--ns:nr=6",
        "--ns:nphi=4",
        "--ns:nz=4",
        "--ns:u0=1.25",
        "--ns:Re="+std::to_string(reynolds),
        "--ns:dt=0.001",
        "--ns:zperiod=1",
        "--ns:verbose=0"
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    config.rewrite(static_cast<int>(argv.size()), argv.data());
    return config;
}

template<typename Function>
bool throws_exception(Function&& function) {
    try {
        function();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void assert_metadata_equal(const fdm::NSCylCheckpointMetadata& actual,
                           const fdm::NSCylCheckpointMetadata& expected) {
    assert_int_equal(actual.schema_version, expected.schema_version);
    assert_string_equal(actual.format_name.c_str(), expected.format_name.c_str());
    assert_int_equal(actual.step_operator_version,
                     expected.step_operator_version);
    assert_string_equal(actual.scalar_type.c_str(), expected.scalar_type.c_str());
    assert_string_equal(actual.axial_boundary.c_str(),
                        expected.axial_boundary.c_str());
    assert_string_equal(actual.state_layout.c_str(), expected.state_layout.c_str());
    assert_string_equal(actual.pressure_gauge.c_str(),
                        expected.pressure_gauge.c_str());
    assert_string_equal(actual.config_text.c_str(), expected.config_text.c_str());
    assert_int_equal(actual.nr, expected.nr);
    assert_int_equal(actual.nphi, expected.nphi);
    assert_int_equal(actual.nz, expected.nz);
    assert_int_equal(actual.state_size, expected.state_size);
    assert_int_equal(actual.u_offset, expected.u_offset);
    assert_int_equal(actual.v_offset, expected.v_offset);
    assert_int_equal(actual.w_offset, expected.w_offset);
    assert_int_equal(actual.p_offset, expected.p_offset);
    assert_int_equal(actual.time_index, expected.time_index);
    assert_true(actual.r == expected.r);
    assert_true(actual.R == expected.R);
    assert_true(actual.h1 == expected.h1);
    assert_true(actual.h2 == expected.h2);
    assert_true(actual.reynolds == expected.reynolds);
    assert_true(actual.dt == expected.dt);
    assert_true(actual.wall_speed == expected.wall_speed);
    assert_true(actual.physical_time == expected.physical_time);
}

template<typename T>
void normalize_pressure(const fdm::NSCylCheckpointMetadata& metadata,
                        std::vector<T>& state) {
    const long double dr = (metadata.R-metadata.r)/metadata.nr;
    long double sum = 0;
    long double weight = 0;
    for (int index = metadata.p_offset; index < metadata.state_size; ++index) {
        const int j = (index-metadata.p_offset)%metadata.nr;
        const long double radius = metadata.r+(j+0.5L)*dr;
        sum += radius*static_cast<long double>(state[index]);
        weight += radius;
    }
    const T mean = static_cast<T>(sum/weight);
    for (int index = metadata.p_offset; index < metadata.state_size; ++index) {
        state[index] -= mean;
    }
}

template<typename T>
void check_round_trip() {
    const Config config = make_config();
    const auto metadata =
        fdm::make_ns_cyl_checkpoint_metadata<T>(config, 123);
    std::vector<T> expected(metadata.state_size);
    for (int i = 0; i < metadata.state_size; ++i) {
        expected[i] = static_cast<T>(0.25*std::sin(0.17*(i+1)));
    }
    normalize_pressure(metadata, expected);
    const std::string filename = std::is_same_v<T, float>
        ? test_file("ns_cyl_checkpoint_float.nc")
        : test_file("ns_cyl_checkpoint_double.nc");
    const fdm::NSCylCheckpointStorage storage(filename);
    storage.save(expected, metadata);

    std::vector<T> actual;
    fdm::NSCylCheckpointMetadata actual_metadata;
    storage.load(actual, actual_metadata,
                 fdm::make_ns_cyl_checkpoint_metadata<T>(config, 0));
    assert_true(actual == expected);
    assert_metadata_equal(actual_metadata, metadata);
}

void test_round_trip_double(void**) {
    check_round_trip<double>();
}

void test_round_trip_float(void**) {
    check_round_trip<float>();
}

void test_rejects_incompatible_or_corrupt_checkpoint(void**) {
    const Config config = make_config();
    const auto metadata =
        fdm::make_ns_cyl_checkpoint_metadata<double>(config, 7);
    const std::string filename = test_file("ns_cyl_checkpoint_invalid.nc");
    const fdm::NSCylCheckpointStorage storage(filename);
    const std::vector<double> zero_state(metadata.state_size, 0.0);
    storage.save(zero_state, metadata);

    std::vector<double> untouched = {3.0, 4.0};
    fdm::NSCylCheckpointMetadata loaded;
    auto incompatible =
        fdm::make_ns_cyl_checkpoint_metadata<double>(make_config(45.0), 0);
    assert_true(throws_exception([&] {
        storage.load(untouched, loaded, incompatible);
    }));
    assert_true((untouched == std::vector<double>{3.0, 4.0}));

    std::vector<float> wrong_type;
    fdm::NSCylCheckpointMetadata wrong_metadata;
    assert_true(throws_exception([&] {
        storage.load(wrong_type, wrong_metadata);
    }));

    int ncid = -1;
    assert_int_equal(nc_open(filename.c_str(), NC_WRITE, &ncid), NC_NOERR);
    int state_variable = -1;
    assert_int_equal(nc_inq_varid(ncid, "state", &state_variable), NC_NOERR);
    const std::size_t pressure_index = metadata.p_offset;
    const double bad_pressure = 1.0;
    assert_int_equal(nc_put_var1_double(
        ncid, state_variable, &pressure_index, &bad_pressure), NC_NOERR);
    assert_int_equal(nc_close(ncid), NC_NOERR);
    assert_true(throws_exception([&] {
        storage.load(untouched, loaded);
    }));
    assert_true((untouched == std::vector<double>{3.0, 4.0}));

    storage.save(zero_state, metadata);

    assert_int_equal(nc_open(filename.c_str(), NC_WRITE, &ncid), NC_NOERR);
    assert_int_equal(nc_redef(ncid), NC_NOERR);
    const int unsupported = 99;
    assert_int_equal(nc_put_att_int(
        ncid, NC_GLOBAL, "schema_version", NC_INT, 1, &unsupported), NC_NOERR);
    assert_int_equal(nc_enddef(ncid), NC_NOERR);
    assert_int_equal(nc_close(ncid), NC_NOERR);
    assert_true(throws_exception([&] {
        storage.load(untouched, loaded);
    }));
}

void test_rejects_invalid_state(void**) {
    const Config config = make_config();
    const auto metadata =
        fdm::make_ns_cyl_checkpoint_metadata<double>(config, 0);
    std::vector<double> state(metadata.state_size, 0);
    state[4] = std::numeric_limits<double>::quiet_NaN();
    const fdm::NSCylCheckpointStorage storage(
        test_file("ns_cyl_checkpoint_non_finite.nc"));
    assert_true(throws_exception([&] {
        storage.save(state, metadata);
    }));

    std::fill(state.begin(), state.end(), 0.0);
    state[metadata.p_offset] = 1.0;
    assert_true(throws_exception([&] {
        storage.save(state, metadata);
    }));
}

void test_restart_reproduces_uninterrupted_trajectory(void**) {
    using Task = fdm::NSCyl<
        double, true, fdm::tensor_flag::periodic>;
    const Config config = make_config();
    Task uninterrupted(config);
    const fdm::NSCylStateLayout<double> layout(uninterrupted);
    layout.initialize_couette_state(uninterrupted);
    for (int i = 0; i < uninterrupted.nphi; ++i) {
        for (int k = 0; k < uninterrupted.nz; ++k) {
            for (int j = 1; j <= uninterrupted.nr; ++j) {
                uninterrupted.v[i][k][j] += 1e-3
                    *std::cos(2*M_PI*i/uninterrupted.nphi)
                    *std::sin(2*M_PI*k/uninterrupted.nz)
                    *std::sin(M_PI*(j-0.5)/uninterrupted.nr);
            }
        }
    }
    uninterrupted.apply_boundary_conditions();
    for (int step = 0; step < 3; ++step) {
        uninterrupted.step();
    }
    auto checkpoint = layout.pack(uninterrupted);
    layout.normalize_packed_pressure(uninterrupted, checkpoint.data());
    assert_true(std::abs(layout.packed_pressure_mean(
                            uninterrupted, checkpoint.data())) < 2e-14);
    layout.unpack(uninterrupted, checkpoint.data());
    const auto metadata = fdm::make_ns_cyl_checkpoint_metadata<double>(
        config, uninterrupted.time_index);
    const fdm::NSCylCheckpointStorage storage(
        test_file("ns_cyl_checkpoint_restart.nc"));
    storage.save(checkpoint, metadata);

    std::vector<double> loaded;
    fdm::NSCylCheckpointMetadata loaded_metadata;
    storage.load(loaded, loaded_metadata,
                 fdm::make_ns_cyl_checkpoint_metadata<double>(config, 0));
    Task resumed(config);
    const fdm::NSCylStateLayout<double> resumed_layout(resumed);
    resumed_layout.unpack(resumed, loaded.data());
    resumed.time_index = loaded_metadata.time_index;
    assert_true(resumed_layout.pack(resumed) == checkpoint);

    for (int step = 0; step < 5; ++step) {
        uninterrupted.step();
        resumed.step();
    }
    const auto expected = layout.pack(uninterrupted);
    const auto actual = resumed_layout.pack(resumed);
    assert_true(actual == expected);
    assert_int_equal(resumed.time_index, uninterrupted.time_index);
}

} // namespace

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_round_trip_double),
        cmocka_unit_test(test_round_trip_float),
        cmocka_unit_test(test_rejects_incompatible_or_corrupt_checkpoint),
        cmocka_unit_test(test_rejects_invalid_state),
        cmocka_unit_test(test_restart_reproduces_uninterrupted_trajectory),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
