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

void test_assignment(void** ) {
    tensor<double, 2> t({0,10,0,10});
    tensor<double, 2> t2({2,8,1,7});

    for (int y = 0; y <= 10; y++) {
        for (int x = 0; x <= 10; x++) {
            t[y][x] = 1;
        }
    }

    for (int y = 2; y <= 8; y++) {
        for (int x = 1; x <= 7; x++) {
            t2[y][x] = 2;
        }
    }

    t = t2;

    for (int y = 0; y <= 10; y++) {
        for (int x = 0; x <= 10; x++) {
            if (2 <= y && y <= 8 &&
                1 <= x && x <= 7)
            {
                assert_int_equal(t[y][x], t2[y][x]);
            } else {
                assert_int_equal(t[y][x], 1);
            }
        }
    }

    for (int y = 2; y <= 8; y++) {
        for (int x = 1; x <= 7; x++) {
            t2[y][x] = 0;
        }
    }

    t2 = t;

    for (int y = 2; y <= 8; y++) {
        for (int x = 1; x <= 7; x++) {
            assert_int_equal(t2[y][x], 2);
        }
    }
}

void test_periodic(void**) {
    tensor<double, 2, true, tensor_flags<tensor_flag::periodic>> t({0, 4, -1, 1});
    t[0][1] = 10;
    t[4][1] = 11;
    verify(t[0][1] == t[5][1]);
    verify(t[-1][1] == t[4][1]);

    tensor<double, 2, true, tensor_flags<tensor_flag::periodic>> t2({-1, 4, -1, 2});
    t2[-1][2] = 20;
    verify(t2[5][2] == t2[-1][2]);
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_read_write_float),
        cmocka_unit_test(test_read_write_double),
        cmocka_unit_test(test_assignment),
        cmocka_unit_test(test_periodic)
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
