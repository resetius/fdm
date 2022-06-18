#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>
#include <random>
#include <vector>

extern "C" {
#include <cmocka.h>
}

#include "mgsch.h"

using namespace std;
using namespace fdm;

template<typename T>
void test_mgsch(bool euclidian) {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-10, 10);

    int n = 100;
    int m = 100;
    vector<vector<T>> vecs;
    vecs.resize(m);

    auto dot1 =  [](const T* x, const T* y, int n) {
        return blas::dot(n, x, 1, y, 1);
    };

    auto dot2 =  [](const T* x, const T* y, int n) {
        T sum = 0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                T r = 0.1 + 0.05*j;
                sum += r*x[i*10+j]*y[i*10+j];
            }
        }
        return sum;
    };

    auto dot = euclidian ? dot1 : dot2;

    for (int k = 0; k < m; k++) {
        vecs[k].resize(n);
        for (int i = 0; i < n; i++) {
            vecs[k][i] = distribution(generator);
        }
    }

    for (int k = 0; k < m; k++) {
        for (int j = 0; j < m; j++) {
            T res = dot(&vecs[k][0], &vecs[j][0], n);
            assert_true(std::fabs(res) > 1e-3);
        }
    }

    mgsch(vecs, m, n, dot);

    T tol = 1e-12;
    if constexpr(is_same<T,float>::value) {
        tol = 1e-3;
    }
    for (int k = 0; k < m; k++) {
        for (int j = 0; j < m; j++) {
            T res = dot(&vecs[k][0], &vecs[j][0], n);
            if (k == j) {
                assert_true(std::fabs(res) > tol);
            } else {
                assert_true(std::fabs(res) < tol);
            }
        }
    }
}

void test_mgsch_double(void**) {
    test_mgsch<double>(true);
}

void test_mgsch_float(void** ) {
    test_mgsch<float>(true);
}

void test_mgsch_cyl_double(void**) {
    test_mgsch<double>(false);
}

void test_mgsch_cyl_float(void** ) {
    test_mgsch<float>(false);
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_mgsch_double, nullptr),
        cmocka_unit_test_prestate(test_mgsch_float, nullptr),
        cmocka_unit_test_prestate(test_mgsch_cyl_double, nullptr),
        cmocka_unit_test_prestate(test_mgsch_cyl_float, nullptr),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
