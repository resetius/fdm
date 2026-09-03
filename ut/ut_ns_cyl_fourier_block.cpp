#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "config.h"
#include "ns_cyl_fourier_block.h"
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
        cmocka_unit_test(test_axisymmetric_block_is_independent_of_nphi),
        cmocka_unit_test(test_dense_spectrum_groups_complex_pair_in_real_columns),
        cmocka_unit_test(test_dense_spectrum_of_real_ns_cyl_block),
        cmocka_unit_test(test_complex_ns_cyl_mode_has_expected_phase_speed),
        cmocka_unit_test(test_spectral_projector_on_block),
        cmocka_unit_test(test_float_block_apply_is_finite_and_nonzero),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
