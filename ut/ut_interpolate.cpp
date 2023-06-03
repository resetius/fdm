#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>
#include <random>

#include "lapl_cyl.h"
#include "config.h"
#include "interpolate.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;
using namespace std;
using namespace std::chrono;
using namespace asp;

template<typename T>
void test_interpolate()
{
    CIC<T,2> int1;
    CIC2<T> int2;
    T l = 10;
    int n = 64;
    T h = l/n;

    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(0, l/2);

    for (int i = 0; i < 100; i++) {
        T x = distribution(generator) + l/4;
        T y = distribution(generator) + l/4;
        int j, k;

        typename CIC2<T>::matrix M;
        int2.distribute(M, x, y, &j, &k, h);

        int jk[] = {0,0};
        T xx[] = {x,y};
        auto M2 = int1.distribute(jk, xx, h);

        assert_int_equal(j, jk[0]);
        assert_int_equal(k, jk[1]);

        for (int k0 = 0; k0 < 2; k0++) {
            for (int j0 = 0; j0 < 2; j0++) {
                assert_float_equal(M[k0][j0], M2[k0][j0], 1e-15);
            }
        }
    }
}

void test_interpolate_double(void** ) {
    test_interpolate<double>();
}

void test_interpolate_float(void** ) {
    test_interpolate<float>();
}

template<typename T>
void test_CIC3() {
    using I = CIC3<T>;
    I interpolator;
    T m = 10.0;
    typename I::matrix M;
    T h = 1.0;
    int i0, k0, j0;

    interpolator.distribute(
        M, 0.5, 0.5, 0.5,
        &j0, &k0, &i0, h
        );
    assert_int_equal(i0, 0);
    assert_int_equal(k0, 0);
    assert_int_equal(j0, 0);
    T f[2][2][2] = {{{0}}};
    for (int i = 0; i < I::n; i++) {
        for (int k = 0; k < I::n; k++) {
            for (int j = 0; j < I::n; j++) {
                f[i+i0][k+k0][j+j0] += m * M[i][k][j];
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        for (int k = 0; k < 2; k++) {
            for (int j = 0; j < 2; j++) {
                assert_float_equal(f[i][k][j], m/8.0, 1e-12);
            }
        }
    }

    interpolator.distribute(
        M, 0.5, 0.6, 0.7,
        &j0, &k0, &i0, h
        );
    assert_int_equal(i0, 0);
    assert_int_equal(k0, 0);
    assert_int_equal(j0, 0);
    T mm = 0;
    for (int i = 0; i < I::n; i++) {
        for (int k = 0; k < I::n; k++) {
            for (int j = 0; j < I::n; j++) {
                mm += m * M[i][k][j];
            }
        }
    }
    assert_float_equal(mm, m, 1e-12);
}

void test_CIC3_double(void** ) {
    test_CIC3<double>();
}

void test_CIC3_float(void** ) {
    test_CIC3<float>();
}

int main(int argc, char** argv) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_interpolate_double, nullptr),
        cmocka_unit_test_prestate(test_interpolate_float, nullptr),
        cmocka_unit_test_prestate(test_CIC3_double, nullptr),
        cmocka_unit_test_prestate(test_CIC3_float, nullptr),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
