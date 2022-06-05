#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <random>
#include <vector>
#include <type_traits>

#include "asp_fft.h"
#include "fft.h"

extern "C" {
#include <cmocka.h>
}

using namespace std;
using namespace fdm;

void test_periodic(void**) {
    using T = double;
    int N = 1024;
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<double> S(N);
    vector<double> s(N);
    vector<double> s1(N);
    fft* ft = FFT_init(FFT_PERIODIC, N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    pFFT_2(&S[0], &s[0], 1.0, ft);
    pFFT_2_1(&s1[0], &S[0], 2.0/N, ft);
    for (int i = 0; i < N; i++) {
        assert_float_equal(s1[i], s[i], 1e-15);
    }

    FFT_free(ft);
}

template<typename T>
void test_periodic_new(void**) {
    int N = 1024;
    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-5;
    }
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    ft.pFFT(&S[0], &s[0], 1.0);
    ft.pFFT_1(&s1[0], &S[0], 2.0/N);
    for (int i = 0; i < N; i++) {
        assert_float_equal(s1[i], s[i], tol);
    }
}

void test_periodic_new_double(void**) {
    test_periodic_new<double>(nullptr);
}

void test_periodic_new_float(void**) {
    test_periodic_new<float>(nullptr);
}

void test_periodic_new_old_cmp(void**) {
    using T = double;
    int N = 1024;
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<double> S(N);
    vector<double> s(N);
    vector<double> S1(N);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fft* ft1 = FFT_init(FFT_PERIODIC, N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    ft.pFFT(&S[0], &s[0], 1.0);
    pFFT_2(&S1[0], &s[0], 1.0, ft1);
    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], 1e-15);
    }

    FFT_free(ft1);
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_periodic),
        cmocka_unit_test(test_periodic_new_double),
        cmocka_unit_test(test_periodic_new_float),
        cmocka_unit_test(test_periodic_new_old_cmp),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
};
