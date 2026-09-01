#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "blas.h"
#include "config.h"
#include "ns_cyl_fourier_block.h"
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

    // матрица блока по столбцам, column major как хочет geev
    std::vector<double> a(static_cast<std::size_t>(n)*n);
    {
        std::vector<double> basis(n, 0.0);
        std::vector<double> image(n);
        for (int column = 0; column < n; ++column) {
            basis[column] = 1;
            block.apply(image.data(), basis.data());
            basis[column] = 0;
            for (int row = 0; row < n; ++row) {
                a[static_cast<std::size_t>(column)*n+row] = image[row];
            }
        }
    }

    std::vector<double> factored(a);   // geev разрушает матрицу
    std::vector<double> wr(n), wi(n);
    std::vector<double> vl(static_cast<std::size_t>(n)*n);
    std::vector<double> vr(static_cast<std::size_t>(n)*n);
    std::vector<double> work(8*n);
    int info = 0;
    fdm::lapack::geev("V", "V", n, factored.data(), n, wr.data(), wi.data(),
                      vl.data(), n, vr.data(), n, work.data(), 8*n, &info);
    assert_int_equal(info, 0);

    // вещественное значение -- один столбец, сопряженная пара -- два
    struct Group {
        int first;
        int count;
        double magnitude;
    };
    std::vector<Group> groups;
    for (int i = 0; i < n; ) {
        const int count = (wi[i] == 0.0) ? 1 : 2;
        groups.push_back({i, count, std::hypot(wr[i], wi[i])});
        i += count;
    }
    std::sort(groups.begin(), groups.end(), [](const Group& x, const Group& y) {
        return x.magnitude > y.magnitude;
    });

    // старшие по модулю группы играют роль неустойчивого подпространства
    auto column_of = [&](const std::vector<double>& v, int column) {
        return std::vector<double>(v.begin()+static_cast<std::size_t>(column)*n,
                                   v.begin()+static_cast<std::size_t>(column+1)*n);
    };

    std::vector<std::vector<double>> e, et;
    std::size_t used = 0;
    while (used < groups.size() && e.size() < 4) {
        const Group& group = groups[used++];
        for (int k = 0; k < group.count; ++k) {
            e.push_back(column_of(vr, group.first+k));
            et.push_back(column_of(vl, group.first+k));
        }
    }
    const int m = static_cast<int>(e.size());
    assert_true(m >= 2);
    assert_true(used < groups.size());

    // базисы биортогональны, матрица Грама не вырождена
    std::vector<double> ete(static_cast<std::size_t>(m)*m);
    const double pivot = fdm::inverse_gramm_matrix(ete.data(), e, et, m, n);
    assert_true(pivot > 1e-8);

    auto project = [&](const std::vector<double>& h) {
        std::vector<double> result(n);
        fdm::projection2(result.data(), h.data(), e, et, ete.data(), m, n);
        return result;
    };

    // geev нормирует собственные вектора, так что абсолютный допуск уместен.
    // Замеренные невязки на этой сетке порядка 1e-16, запас четыре порядка.
    const double tol = 1e-12;

    // P r_j = r_j
    for (int i = 0; i < m; ++i) {
        const auto projected = project(e[i]);
        for (int k = 0; k < n; ++k) {
            assert_true(std::abs(projected[k]-e[i][k]) < tol);
        }
    }

    // P r = 0 для вектора из дополнительного инвариантного подпространства
    {
        const auto outside = column_of(vr, groups[used].first);
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

    // P*P = P и P+ + P- = I
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
        cmocka_unit_test(test_linear_step_preserves_real_packed_block),
        cmocka_unit_test(test_spectral_projector_on_block),
        cmocka_unit_test(test_float_block_apply_is_finite_and_nonzero),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
