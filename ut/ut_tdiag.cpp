#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <random>
#include <vector>
#include <type_traits>
#include <chrono>

#include "asp_gauss.h"
#include "blas.h"
#include "config.h"
#include "cyclic_reduction.h"

extern "C" {
#include <cmocka.h>
}

using namespace std;
using namespace std::chrono;
using namespace fdm;

template<typename T>
void test_gtsv(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 255);
    int verbose = c->get("test", "verbose", 0);
    vector<T> A1(N);
    vector<T> A2(N);
    vector<T> A3(N);
    vector<T> B(N);
    vector<T> B1(N);
    vector<T> B2(N);

    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    for (int i = 0; i < N; i++) {
        A1[i] = distribution(generator);
        A2[i] = distribution(generator);
        A3[i] = distribution(generator);
        B[i] = distribution(generator);
    }

    {
        auto A11 = A1;
        auto A21 = A2;
        auto A31 = A3;
        B1 = B;

        auto t1 = steady_clock::now();
        if constexpr(std::is_same_v<T, double>) {
            solve_tdiag_linear_my(B1.data(), A11.data(), A21.data(), A31.data(), N);
        } else {
            solve_tdiag_linearf_my(B1.data(), A11.data(), A21.data(), A31.data(), N);
        }
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t1=%f\n", interval.count());
        }
    }

    {
        auto A11 = A1;
        auto A21 = A2;
        auto A31 = A3;
        B2 = B;
        int info = 0;
        auto t1 = steady_clock::now();
        lapack::gtsv(N, 1, A11.data(), A21.data(), A31.data(), B2.data(), N, &info);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t1=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(B1[i], B2[i], tol);
        if (verbose > 1) {
            printf("%e <> %e\n", B1[i], B2[i]);
        }
    }
}

void test_gtsv_double(void** s) {
    test_gtsv<double>(s);
}

void test_gtsv_float(void** s) {
    test_gtsv<float>(s);
}

// for power of two
template<typename T>
void test_cr(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 31);
    int verbose = c->get("test", "verbose", 0);
    vector<T> A1(N+1);
    vector<T> A2(N);
    vector<T> A3(N);
    vector<T> B(N);
    vector<T> B1(N);
    vector<T> B2(N);

    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    for (int i = 0; i < N; i++) {
        A1[i] = distribution(generator);
        A2[i] = distribution(generator);
        A3[i] = distribution(generator);
        B[i] = distribution(generator);
    }

    {
        auto A11 = A1;
        auto A21 = A2;
        auto A31 = A3;
        B1 = B;

        auto t1 = steady_clock::now();
        if constexpr(std::is_same_v<T, double>) {
            solve_tdiag_linear_my(B1.data(), A11.data(), A21.data(), A31.data(), N);
        } else {
            solve_tdiag_linearf_my(B1.data(), A11.data(), A21.data(), A31.data(), N);
        }
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t1=%f\n", interval.count());
        }
    }

    {
        auto A11 = A1;
        auto A21 = A2;
        auto A31 = A3;
        B2 = B;
        auto t1 = steady_clock::now();
        int q = ceil(log2(N+1));
        cyclic_reduction(A21.data(), A11.data(), A31.data(), B2.data(), q, N);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t1=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(B1[i], B2[i], tol);
        if (verbose > 1) {
            printf("%e <> %e\n", B1[i], B2[i]);
        }
    }
}

void test_cr_double(void** s) {
    test_cr<double>(s);
}

void test_cr_float(void** s) {
    test_cr<float>(s);
}

// for generic case
template<typename T>
void test_crg(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 32);
    int verbose = c->get("test", "verbose", 0);
    vector<T> A1(N+1);
    vector<T> A2(N);
    vector<T> A3(N);
    vector<T> B(N);
    vector<T> B1(N);
    vector<T> B2(N);

    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    for (int i = 0; i < N; i++) {
        A1[i] = distribution(generator);
        A2[i] = distribution(generator);
        A3[i] = distribution(generator);
        B[i] = distribution(generator);
    }

    {
        auto A11 = A1;
        auto A21 = A2;
        auto A31 = A3;
        B1 = B;

        auto t1 = steady_clock::now();
        if constexpr(std::is_same_v<T, double>) {
            solve_tdiag_linear_my(B1.data(), A11.data(), A21.data(), A31.data(), N);
        } else {
            solve_tdiag_linearf_my(B1.data(), A11.data(), A21.data(), A31.data(), N);
        }
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t1=%f\n", interval.count());
        }
    }

    {
        auto A11 = A1;
        auto A21 = A2;
        auto A31 = A3;
        B2 = B;
        auto t1 = steady_clock::now();
        int q = ceil(log2(N+1));
        cyclic_reduction_general(A21.data(), A11.data(), A31.data(), B2.data(), q, N);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t1=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(B1[i], B2[i], tol);
        if (verbose > 1) {
            printf("%e <> %e\n", B1[i], B2[i]);
        }
    }
}

void test_crg_double(void** s) {
    test_cr<double>(s);
}

void test_crg_float(void** s) {
    test_cr<float>(s);
}

int main(int argc, char** argv) {
    string config_fn = "ut_tdiag.ini";
    Config c;
    c.open(config_fn);
    c.rewrite(argc, argv);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_gtsv_float, &c),
        cmocka_unit_test_prestate(test_gtsv_double, &c),
        cmocka_unit_test_prestate(test_cr_float, &c),
        cmocka_unit_test_prestate(test_cr_double, &c),
        cmocka_unit_test_prestate(test_crg_float, &c),
        cmocka_unit_test_prestate(test_crg_double, &c),
    };

    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
