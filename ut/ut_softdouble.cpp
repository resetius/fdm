#include "soft_double.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <random>
#include <vector>
#include <type_traits>
#include <chrono>
#include <iostream>

#include <complex>

extern "C" {
#include <cmocka.h>
}

void test_conversion(void** s) {
    SoftDouble sd1(1.0f);
    SoftDouble sd2(2.0f);

    float f1 = static_cast<float>(sd1);
    float f2 = static_cast<float>(sd2);

    assert_float_equal(f1, 1.0f, 1e-6);
    assert_float_equal(f2, 2.0f, 1e-6);
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_conversion),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
