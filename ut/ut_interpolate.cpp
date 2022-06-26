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

int main(int argc, char** argv) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_interpolate_double, nullptr),
        cmocka_unit_test_prestate(test_interpolate_float, nullptr),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
