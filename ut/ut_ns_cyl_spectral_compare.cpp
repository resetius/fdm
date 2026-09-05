#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <cmath>
#include <vector>

#include "ns_cyl_spectral_compare.h"
#include "ns_cyl_state.h"

extern "C" {
#include <cmocka.h>
}

namespace {

fdm::NSCylSpectralMetadata metadata(int nr) {
    fdm::NSCylSpectralMetadata result;
    result.scalar_type = "float64";
    result.nr = nr;
    result.nphi = nr;
    result.nz = nr;
    result.radial_size = 4*nr-1;
    result.u_offset = 0;
    result.v_offset = nr-1;
    result.w_offset = 2*nr-1;
    result.p_offset = 3*nr-1;
    result.operator_steps = 1;
    result.r = 1;
    result.R = 2;
    result.h1 = 0;
    result.h2 = 2;
    result.reynolds = 100;
    result.dt = 0.001;
    result.wall_speed = 1;
    result.growth_tolerance = 1e-8;
    result.residual_tolerance = 1e-10;
    result.condition_limit = 1e10;
    return result;
}

double normalized_radius(double radius) {
    return radius-1;
}

double u_profile(double x) {
    return x <= 0.5 ? 2*x : 2*(1-x);
}

double cell_profile(double x) {
    constexpr double peak = 0.375;
    return x <= peak ? x/peak : (1-x)/(1-peak);
}

void fill_staggered_column(std::vector<double>& column,
                           const fdm::NSCylSpectralMetadata& grid,
                           int phase_count) {
    using Layout = fdm::NSCylStateLayout<double>;
    using Component = Layout::Component;
    const Layout layout(grid.nr, grid.nz, grid.nphi);
    const double dr = (grid.R-grid.r)/grid.nr;
    for (int phase = 0; phase < phase_count; ++phase) {
        const double scale = phase+1;
        const int offset = phase*layout.radial_size;
        for (int j = 1; j < grid.nr; ++j) {
            const double x = normalized_radius(grid.r+j*dr);
            column[offset+layout.radial_index(Component::u, j)] =
                scale*u_profile(x);
        }
        for (int j = 1; j <= grid.nr; ++j) {
            const double x = normalized_radius(grid.r+(j-0.5)*dr);
            column[offset+layout.radial_index(Component::v, j)] =
                scale*cell_profile(x);
            column[offset+layout.radial_index(Component::w, j)] =
                -2*scale*cell_profile(x);
            column[offset+layout.radial_index(Component::p, j)] =
                scale*(2+3*x);
        }
    }
}

void test_staggered_radial_prolongation(void**) {
    using Layout = fdm::NSCylStateLayout<double>;
    using Component = Layout::Component;
    const auto coarse = metadata(4);
    const auto fine = metadata(8);
    constexpr int phases = 2;
    const Layout coarse_layout(coarse.nr, coarse.nz, coarse.nphi);
    const Layout fine_layout(fine.nr, fine.nz, fine.nphi);
    std::vector<double> source(phases*coarse_layout.radial_size, 0);
    fill_staggered_column(source, coarse, phases);

    const auto actual = fdm::prolong_ns_cyl_spectral_column(
        source.data(), static_cast<int>(source.size()), phases, false,
        coarse, fine);
    assert_int_equal(actual.size(), phases*fine_layout.radial_size);
    const double dr = (fine.R-fine.r)/fine.nr;
    for (int phase = 0; phase < phases; ++phase) {
        const double scale = phase+1;
        const int offset = phase*fine_layout.radial_size;
        for (int j = 1; j < fine.nr; ++j) {
            const double x = normalized_radius(fine.r+j*dr);
            assert_true(std::abs(actual[offset+fine_layout.radial_index(
                Component::u, j)]-scale*u_profile(x)) < 2e-15);
        }
        for (int j = 1; j <= fine.nr; ++j) {
            const double x = normalized_radius(fine.r+(j-0.5)*dr);
            assert_true(std::abs(actual[offset+fine_layout.radial_index(
                Component::v, j)]-scale*cell_profile(x)) < 2e-15);
            assert_true(std::abs(actual[offset+fine_layout.radial_index(
                Component::w, j)]+2*scale*cell_profile(x)) < 4e-15);
            assert_true(std::abs(actual[offset+fine_layout.radial_index(
                Component::p, j)]-scale*(2+3*x)) < 2e-15);
        }
    }

    const auto identity = fdm::prolong_ns_cyl_spectral_column(
        source.data(), static_cast<int>(source.size()), phases, false,
        coarse, coarse);
    assert_int_equal(identity.size(), source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        assert_true(std::abs(identity[i]-source[i]) < 2e-15);
    }
}

void test_zero_gauge_survives_prolongation(void**) {
    using Layout = fdm::NSCylStateLayout<double>;
    using Component = Layout::Component;
    const auto coarse = metadata(4);
    const auto fine = metadata(8);
    const Layout coarse_layout(coarse.nr, coarse.nz, coarse.nphi);
    const Layout fine_layout(fine.nr, fine.nz, fine.nphi);
    std::vector<double> source(coarse_layout.zero_gauge_block_size(), 0);
    for (int j = 1; j < coarse.nr; ++j) {
        source[coarse_layout.radial_index(Component::p, j)] = 0.4*j-0.7;
    }
    const auto prolonged = fdm::prolong_ns_cyl_spectral_column(
        source.data(), static_cast<int>(source.size()), 1, true,
        coarse, fine);
    assert_int_equal(prolonged.size(), fine_layout.zero_gauge_block_size());

    std::vector<double> full(fine_layout.radial_size, 0);
    std::copy(prolonged.begin(), prolonged.end(), full.begin());
    const double dr = (fine.R-fine.r)/fine.nr;
    long double partial_sum = 0;
    for (int j = 1; j < fine.nr; ++j) {
        const double radius = fine.r+(j-0.5)*dr;
        partial_sum += radius*full[fine_layout.radial_index(Component::p, j)];
    }
    const double last_radius = fine.r+(fine.nr-0.5)*dr;
    full[fine_layout.radial_index(Component::p, fine.nr)] =
        static_cast<double>(-partial_sum/last_radius);
    long double weighted_sum = 0;
    long double weight = 0;
    for (int j = 1; j <= fine.nr; ++j) {
        const double radius = fine.r+(j-0.5)*dr;
        weighted_sum += radius*full[fine_layout.radial_index(Component::p, j)];
        weight += radius;
    }
    assert_true(std::abs(static_cast<double>(weighted_sum/weight)) < 2e-16);
}

void test_principal_angle_is_basis_invariant(void**) {
    const double inverse_sqrt_two = 1/std::sqrt(2.0);
    const std::vector<std::vector<double>> coordinate = {
        {1, 0, 0, 0},
        {0, 1, 0, 0}
    };
    const std::vector<std::vector<double>> rotated = {
        {inverse_sqrt_two, inverse_sqrt_two, 0, 0},
        {-inverse_sqrt_two, inverse_sqrt_two, 0, 0}
    };
    const std::vector<std::vector<double>> orthogonal = {
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    assert_true(fdm::ns_cyl_subspace_maximum_sine(coordinate, rotated)
                < 2e-15);
    assert_true(std::abs(fdm::ns_cyl_subspace_maximum_sine(
        coordinate, orthogonal)-1) < 2e-15);

    const double angle = 0.37;
    const std::vector<std::vector<double>> line = {{1, 0}};
    const std::vector<std::vector<double>> tilted = {
        {std::cos(angle), std::sin(angle)}
    };
    assert_true(std::abs(fdm::ns_cyl_subspace_maximum_sine(line, tilted)
                         -std::sin(angle)) < 2e-15);
}

} // namespace

int main() {
    const CMUnitTest tests[] = {
        cmocka_unit_test(test_staggered_radial_prolongation),
        cmocka_unit_test(test_zero_gauge_survives_prolongation),
        cmocka_unit_test(test_principal_angle_is_basis_invariant)
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
