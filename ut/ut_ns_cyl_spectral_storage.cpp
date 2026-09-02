#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <netcdf.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "config.h"
#include "eigenvectors_storage.h"
#include "ns_cyl_spectral_projector.h"
#include "ns_cyl_spectral_storage.h"

extern "C" {
#include <cmocka.h>
}

namespace {

std::string test_file(const char* name) {
    return std::string(NS_CYL_SPECTRAL_TEST_DIR)+"/"+name;
}

Config make_config() {
    Config config;
    std::vector<std::string> arguments = {
        "ut_ns_cyl_spectral_storage",
        "--ns:r=1.0",
        "--ns:R=2.0",
        "--ns:h1=0.0",
        "--ns:h2=6.0",
        "--ns:nr=2",
        "--ns:nphi=4",
        "--ns:nz=4",
        "--ns:u0=1.25",
        "--ns:Re=44.0",
        "--ns:dt=0.1",
        "--spectral:operator_steps=2",
        "--spectral:growth_tol=1e-8",
        "--spectral:residual_tol=1e-5",
        "--spectral:condition_limit=1e6"
    };
    std::vector<char*> argv;
    for (auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    config.rewrite(static_cast<int>(argv.size()), argv.data());
    return config;
}

template<typename T>
fdm::NSCylSpectralMode<T> make_mode(
    const fdm::NSCylSpectralMetadata& metadata, int m, int l,
    std::complex<T> multiplier, int first_basis_column) {
    fdm::NSCylSpectralMode<T> mode;
    mode.m = m;
    mode.l = l;
    const int phi_phases = (m == 0 || 2*m == metadata.nphi) ? 1 : 2;
    const int z_phases = (l == 0 || 2*l == metadata.nz) ? 1 : 2;
    mode.phase_count = phi_phases*z_phases;
    mode.radial_size = metadata.radial_size;
    mode.pressure_gauge_fixed = m == 0 && l == 0;
    mode.block_size = mode.radial_size*mode.phase_count
        -(mode.pressure_gauge_fixed ? 1 : 0);
    mode.multiplier = multiplier;
    const double duration = metadata.operator_steps*metadata.dt;
    mode.growth_rate = std::log(std::abs(multiplier))/duration;
    mode.frequency = std::atan2(
        static_cast<double>(multiplier.imag()),
        static_cast<double>(multiplier.real()))/duration;
    mode.right_residual = 1e-7;
    mode.left_residual = 2e-7;
    mode.growing = true;
    mode.residual_accepted = true;
    mode.column_count = multiplier.imag() == T(0) ? 1 : 2;
    mode.right_columns.assign(
        static_cast<std::size_t>(mode.column_count)*mode.block_size, T(0));
    mode.left_columns.assign(mode.right_columns.size(), T(0));
    for (int column = 0; column < mode.column_count; ++column) {
        const int coordinate = first_basis_column+column;
        mode.right_columns[
            static_cast<std::size_t>(column)*mode.block_size+coordinate] = T(1);
        mode.left_columns[
            static_cast<std::size_t>(column)*mode.block_size+coordinate] = T(1);
    }
    return mode;
}

template<typename T>
fdm::NSCylSpectralModeSet<T> make_modes(
    const fdm::NSCylSpectralMetadata& metadata) {
    fdm::NSCylSpectralModeSet<T> result;
    result.append_filterable_mode(make_mode<T>(
        metadata, 1, 0, {T(1.08), T(0)}, 0));
    result.append_filterable_mode(make_mode<T>(
        metadata, 0, 1, {T(1.05), T(0)}, 2));
    result.append_filterable_mode(make_mode<T>(
        metadata, 0, 1, {T(1.1), T(0.2)}, 0));
    return result;
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

void assert_metadata_equal(const fdm::NSCylSpectralMetadata& actual,
                           const fdm::NSCylSpectralMetadata& expected) {
    assert_int_equal(actual.schema_version, expected.schema_version);
    assert_string_equal(actual.operator_name.c_str(),
                        expected.operator_name.c_str());
    assert_int_equal(actual.operator_version, expected.operator_version);
    assert_string_equal(actual.scalar_type.c_str(), expected.scalar_type.c_str());
    assert_string_equal(actual.fourier_layout.c_str(),
                        expected.fourier_layout.c_str());
    assert_string_equal(actual.state_layout.c_str(),
                        expected.state_layout.c_str());
    assert_string_equal(actual.pressure_gauge.c_str(),
                        expected.pressure_gauge.c_str());
    assert_string_equal(actual.config_text.c_str(), expected.config_text.c_str());
    assert_int_equal(actual.nr, expected.nr);
    assert_int_equal(actual.nphi, expected.nphi);
    assert_int_equal(actual.nz, expected.nz);
    assert_int_equal(actual.radial_size, expected.radial_size);
    assert_int_equal(actual.u_offset, expected.u_offset);
    assert_int_equal(actual.v_offset, expected.v_offset);
    assert_int_equal(actual.w_offset, expected.w_offset);
    assert_int_equal(actual.p_offset, expected.p_offset);
    assert_int_equal(actual.operator_steps, expected.operator_steps);
    assert_true(actual.r == expected.r);
    assert_true(actual.R == expected.R);
    assert_true(actual.h1 == expected.h1);
    assert_true(actual.h2 == expected.h2);
    assert_true(actual.reynolds == expected.reynolds);
    assert_true(actual.dt == expected.dt);
    assert_true(actual.wall_speed == expected.wall_speed);
    assert_true(actual.growth_tolerance == expected.growth_tolerance);
    assert_true(actual.residual_tolerance == expected.residual_tolerance);
    assert_true(actual.condition_limit == expected.condition_limit);
}

template<typename T>
void assert_mode_equal(const fdm::NSCylSpectralMode<T>& actual,
                       const fdm::NSCylSpectralMode<T>& expected) {
    assert_int_equal(actual.m, expected.m);
    assert_int_equal(actual.l, expected.l);
    assert_int_equal(actual.phase_count, expected.phase_count);
    assert_int_equal(actual.radial_size, expected.radial_size);
    assert_int_equal(actual.block_size, expected.block_size);
    assert_int_equal(actual.pressure_gauge_fixed,
                     expected.pressure_gauge_fixed);
    assert_int_equal(actual.column_count, expected.column_count);
    assert_true(actual.multiplier == expected.multiplier);
    assert_true(actual.growth_rate == expected.growth_rate);
    assert_true(actual.frequency == expected.frequency);
    assert_true(actual.right_residual == expected.right_residual);
    assert_true(actual.left_residual == expected.left_residual);
    assert_true(actual.growing);
    assert_true(actual.residual_accepted);
    assert_true(actual.right_columns == expected.right_columns);
    assert_true(actual.left_columns == expected.left_columns);
    assert_true(std::isfinite(actual.block_condition_number));
    assert_true(actual.block_condition_number >= 1.0);
}

template<typename T>
void test_round_trip() {
    const Config config = make_config();
    const auto metadata = fdm::make_ns_cyl_spectral_metadata<T>(config);
    auto expected_modes = make_modes<T>(metadata);
    expected_modes.sort_by_block_and_growth();
    const std::string filename = std::is_same_v<T, float>
        ? test_file("ns_cyl_spectrum_float.nc")
        : test_file("ns_cyl_spectrum_double.nc");
    const fdm::NSCylSpectralStorage storage(filename);
    storage.save(make_modes<T>(metadata), metadata);

    fdm::NSCylSpectralModeSet<T> loaded_modes;
    fdm::NSCylSpectralMetadata loaded_metadata;
    storage.load(loaded_modes, loaded_metadata, metadata);
    assert_metadata_equal(loaded_metadata, metadata);
    assert_int_equal(loaded_modes.size(), expected_modes.size());
    assert_int_equal(loaded_modes.real_dimension(), 4);
    for (int i = 0; i < static_cast<int>(loaded_modes.size()); ++i) {
        assert_mode_equal(loaded_modes.modes()[i], expected_modes.modes()[i]);
    }
    assert_int_equal(loaded_modes.modes()[0].m, 0);
    assert_int_equal(loaded_modes.modes()[0].l, 1);
    assert_int_equal(loaded_modes.modes()[0].column_count, 2);
    assert_int_equal(loaded_modes.modes()[1].m, 0);
    assert_int_equal(loaded_modes.modes()[1].column_count, 1);
    assert_int_equal(loaded_modes.modes()[2].m, 1);

    const fdm::NSCylSpectralProjector<T> projector(loaded_modes, 1e6);
    assert_int_equal(projector.blocks().size(), 2);
    assert_int_equal(projector.real_dimension(), 4);

    auto incompatible = metadata;
    incompatible.dt *= 2;
    fdm::NSCylSpectralModeSet<T> untouched = make_modes<T>(metadata);
    const std::size_t original_size = untouched.size();
    assert_true(throws_exception([&] {
        storage.load(untouched, loaded_metadata, incompatible);
    }));
    assert_int_equal(untouched.size(), original_size);
}

void test_rejects_incompatible_metadata(void**) {
    const auto metadata =
        fdm::make_ns_cyl_spectral_metadata<double>(make_config());
    const fdm::NSCylSpectralStorage storage(
        test_file("ns_cyl_spectrum_incompatible.nc"));
    storage.save(make_modes<double>(metadata), metadata);

    const auto rejects = [&](fdm::NSCylSpectralMetadata expected) {
        fdm::NSCylSpectralModeSet<double> modes;
        fdm::NSCylSpectralMetadata loaded;
        assert_true(throws_exception([&] {
            storage.load(modes, loaded, expected);
        }));
    };

    auto expected = metadata;
    expected.nphi += 2;
    rejects(expected);
    expected = metadata;
    expected.reynolds += 1;
    rejects(expected);
    expected = metadata;
    expected.dt *= 2;
    rejects(expected);
    expected = metadata;
    expected.state_layout += "_incompatible";
    rejects(expected);
}

void test_round_trip_double(void**) {
    test_round_trip<double>();
}

void test_round_trip_float(void**) {
    test_round_trip<float>();
}

void test_empty_mode_set(void**) {
    const auto metadata =
        fdm::make_ns_cyl_spectral_metadata<double>(make_config());
    const fdm::NSCylSpectralModeSet<double> empty;
    const fdm::NSCylSpectralStorage storage(
        test_file("ns_cyl_spectrum_empty.nc"));
    storage.save(empty, metadata);

    fdm::NSCylSpectralModeSet<double> loaded;
    fdm::NSCylSpectralMetadata loaded_metadata;
    storage.load(loaded, loaded_metadata, metadata);
    assert_true(loaded.empty());
    assert_int_equal(loaded.real_dimension(), 0);
    const fdm::NSCylSpectralProjector<double> projector(loaded, 1e6);
    assert_true(projector.blocks().empty());
}

void test_rejects_wrong_scalar_type_and_legacy_file(void**) {
    const Config config = make_config();
    const auto float_metadata =
        fdm::make_ns_cyl_spectral_metadata<float>(config);
    const fdm::NSCylSpectralStorage float_storage(
        test_file("ns_cyl_spectrum_wrong_type.nc"));
    float_storage.save(make_modes<float>(float_metadata), float_metadata);
    fdm::NSCylSpectralModeSet<double> wrong_type;
    fdm::NSCylSpectralMetadata loaded_metadata;
    assert_true(throws_exception([&] {
        float_storage.load(wrong_type, loaded_metadata);
    }));

    const std::string legacy_filename =
        test_file("ns_cyl_spectrum_legacy.nc");
    fdm::eigenvectors_storage legacy(legacy_filename);
    std::vector<std::vector<double>> vectors = {{1.0, 2.0, 3.0}};
    legacy.save(vectors, {0}, config);
    const fdm::NSCylSpectralStorage new_storage(legacy_filename);
    assert_true(throws_exception([&] {
        new_storage.load(wrong_type, loaded_metadata);
    }));
}

void test_rejects_corrupt_column_layout(void**) {
    const auto metadata =
        fdm::make_ns_cyl_spectral_metadata<double>(make_config());
    const std::string filename = test_file("ns_cyl_spectrum_corrupt.nc");
    const fdm::NSCylSpectralStorage storage(filename);
    storage.save(make_modes<double>(metadata), metadata);

    int ncid = -1;
    assert_int_equal(nc_open(filename.c_str(), NC_WRITE, &ncid), NC_NOERR);
    int variable = -1;
    assert_int_equal(nc_inq_varid(ncid, "column_count", &variable), NC_NOERR);
    const std::size_t index[] = {0};
    const int invalid_column_count = 3;
    assert_int_equal(nc_put_var1_int(
        ncid, variable, index, &invalid_column_count), NC_NOERR);
    assert_int_equal(nc_close(ncid), NC_NOERR);

    fdm::NSCylSpectralModeSet<double> loaded;
    fdm::NSCylSpectralMetadata loaded_metadata;
    assert_true(throws_exception([&] {
        storage.load(loaded, loaded_metadata, metadata);
    }));
}

void test_rejects_unknown_schema(void**) {
    const auto metadata =
        fdm::make_ns_cyl_spectral_metadata<double>(make_config());
    const std::string filename = test_file("ns_cyl_spectrum_schema.nc");
    const fdm::NSCylSpectralStorage storage(filename);
    storage.save(make_modes<double>(metadata), metadata);

    int ncid = -1;
    assert_int_equal(nc_open(filename.c_str(), NC_WRITE, &ncid), NC_NOERR);
    assert_int_equal(nc_redef(ncid), NC_NOERR);
    const int unsupported_version = 99;
    assert_int_equal(nc_put_att_int(
        ncid, NC_GLOBAL, "schema_version", NC_INT, 1,
        &unsupported_version), NC_NOERR);
    assert_int_equal(nc_enddef(ncid), NC_NOERR);
    assert_int_equal(nc_close(ncid), NC_NOERR);

    fdm::NSCylSpectralModeSet<double> loaded;
    fdm::NSCylSpectralMetadata loaded_metadata;
    assert_true(throws_exception([&] {
        storage.load(loaded, loaded_metadata, metadata);
    }));
}

} // namespace

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_round_trip_double),
        cmocka_unit_test(test_round_trip_float),
        cmocka_unit_test(test_rejects_incompatible_metadata),
        cmocka_unit_test(test_empty_mode_set),
        cmocka_unit_test(test_rejects_wrong_scalar_type_and_legacy_file),
        cmocka_unit_test(test_rejects_corrupt_column_layout),
        cmocka_unit_test(test_rejects_unknown_schema),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
