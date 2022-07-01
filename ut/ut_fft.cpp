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

    s1 = s; // s1 will be utilized by pFFT
    ft.pFFT(&S[0], &s1[0], 1.0);
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

template<typename T>
void test_sin(void**) {
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
    vector<T> s2(N);

    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    ft.sFFT(&S[0], &s[0], 1.0);
    ft.sFFT(&s1[0], &S[0], 2.0/N);

    for (int i = 1; i < N-1; i++) {
        assert_float_equal(s1[i], s[i], tol);
    }

    if constexpr(is_same<T,double>::value) {
        sFT(&s2[0], &S[0], 2.0/N, N);
        for (int i = 1; i < N-1; i++) {
            assert_float_equal(s1[i], s2[i], tol);
        }
    }
}

void test_sin_double(void**) {
    test_sin<double>(nullptr);
}

void test_sin_float(void**) {
    test_sin<float>(nullptr);
}

template<typename T>
void test_cos(void**) {
    int N = 8;
    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-5;
    }
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<T> S(N+1);
    vector<T> s(N+1);
    vector<T> s1(N+1);
    vector<T> s2(N+1);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);

    for (int i = 0; i <= N; i++) {
        s[i] = distribution(generator);
    }

    ft.cFFT(&S[0], &s[0], 1.0);

    if constexpr(is_same<T,double>::value) {
        cFT(&s2[0], &s[0], 1.0, N);
        for (int i = 0; i <= N; i++) {
            assert_float_equal(S[i], s2[i], tol);
        }
    }

    ft.cFFT(&s1[0], &S[0], 2.0/N);

    for (int i = 0; i <= N; i++) {
        assert_float_equal(s1[i], s[i], tol);
    }

    if constexpr(is_same<T,double>::value) {
        cFT(&s2[0], &S[0], 2.0/N, N);
        for (int i = 1; i < N-1; i++) {
            assert_float_equal(s1[i], s2[i], tol);
        }
    }
}

void test_cos_double(void**) {
    test_cos<double>(nullptr);
}

void test_cos_float(void**) {
    test_cos<float>(nullptr);
}

void test_periodic_new_old_cmp(void**) {
    using T = double;
    int N = 1024;
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<double> S(N);
    vector<double> s(N);
    vector<double> s1(N);
    vector<double> S1(N);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fft* ft1 = FFT_init(FFT_PERIODIC, N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    ft.pFFT(&S[0], &s1[0], 1.0);
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
        cmocka_unit_test(test_sin_double),
        cmocka_unit_test(test_sin_float),
        cmocka_unit_test(test_cos_double),
        cmocka_unit_test(test_cos_float),
        cmocka_unit_test(test_periodic_new_old_cmp),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
};
