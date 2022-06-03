#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>

#include "tensor.h"

extern "C" {
#include <cmocka.h>
}

using namespace std;
using namespace fdm;

template<typename T, bool check>
void check_read_write() {
    T tol = 1e-7;
    if constexpr(is_same<T,double>::value) {
        tol = 1e-15;
    }

    int x1 = -1, x2 = 10;
    int y1 = -2, y2 = 5;
    int z1 = -5, z2 = 6;
    tensor<T, 3, check> t({z1, z2, y1, y2, x1, x2});
    for (int i = z1; i <= z2; i++) {
        for (int j = y1; j <= y2; j++) {
            for (int k = x1; k <= x2; k++) {
                t[i][j][k] = (i+0.5)*(j+0.5)*(k+0.5);
            }
        }
    }
    for (int i = z1; i <= z2; i++) {
        for (int j = y1; j <= y2; j++) {
            for (int k = x1; k <= x2; k++) {
                assert_float_equal(
                    fabs((i+0.5)*(j+0.5)*(k+0.5)-t[i][j][k]), 0.0, tol
                    );
            }
        }
    }
}

void test_read_write_float(void** ) {
    check_read_write<float, true>();
    check_read_write<float, false>();
}

void test_read_write_double(void** ) {
    check_read_write<double, true>();
    check_read_write<double, false>();
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_read_write_float),
        cmocka_unit_test(test_read_write_double),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
