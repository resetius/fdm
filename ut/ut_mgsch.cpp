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
#include "projection.h"

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

template<typename T>
void test_ortoproj_simple_along(void**) {
    vector<vector<T>> basis = {
        {1,0,0,0},
        {0,1,0,0}
    };

    vector<T> vec = {1,2,3,4};
    ortoproj_along(&vec[0], basis, 2, 4);
    assert_float_equal(vec[2], 3, 1e-15);
    assert_float_equal(vec[3], 4, 1e-15);
}

void test_ortoproj_simple_along_float(void** data) {
    test_ortoproj_simple_along<float>(data);
}

void test_ortoproj_simple_along_double(void** data) {
    test_ortoproj_simple_along<double>(data);
}

template<typename T>
void test_ortoproj_along(void**) {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-10, 10);

    int n = 20;
    int m = 100;
    vector<vector<T>> vecs;
    vecs.resize(n);

    for (int k = 0; k < n; k++) {
        vecs[k].resize(m);
        for (int i = 0; i < m; i++) {
            vecs[k][i] = distribution(generator);
        }
    }

    mgsch<T>(vecs, n, m);

    vector<T> vec(m);
    for (int i = 0; i < m; i++) {
        vec[i] = distribution(generator);
    }
    ortoproj_along(&vec[0], vecs, n, m);
    vector<T> t = vec;
    ortoproj_along(&vec[0], vecs, n, m);
    T tol = 1e-15;
    if constexpr(is_same<T,float>::value) {
        tol = 1e-4;
    }
    for (int i = 0; i < m; i++) {
        assert_float_equal(vec[i], t[i], tol);
    }
}

void test_ortoproj_along_float(void** data) {
    test_ortoproj_along<float>(data);
}

void test_ortoproj_along_double(void** data) {
    test_ortoproj_along<double>(data);
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_mgsch_double, nullptr),
        cmocka_unit_test_prestate(test_mgsch_float, nullptr),
        cmocka_unit_test_prestate(test_mgsch_cyl_double, nullptr),
        cmocka_unit_test_prestate(test_mgsch_cyl_float, nullptr),
        cmocka_unit_test_prestate(test_ortoproj_simple_along_float, nullptr),
        cmocka_unit_test_prestate(test_ortoproj_simple_along_double, nullptr),
        cmocka_unit_test_prestate(test_ortoproj_along_float, nullptr),
        cmocka_unit_test_prestate(test_ortoproj_along_double, nullptr),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
