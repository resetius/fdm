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
        cmocka_unit_test(test_linear_step_preserves_real_packed_block),
        cmocka_unit_test(test_float_block_apply_is_finite_and_nonzero),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
