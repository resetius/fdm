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

void test_sum(void** s) {
    SoftDouble a(1./2. + 1./4.);
    SoftDouble b(1./2.);
    auto c = a+b;
    double ans = 1./2. + 1./4. + 1/2.;

    assert_float_equal((double)c, ans, 1e-20);

    printf("%lx %lx\n", std::bit_cast<uint64_t>((double)c), std::bit_cast<uint64_t>(ans));
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_conversion),
        cmocka_unit_test(test_sum),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
