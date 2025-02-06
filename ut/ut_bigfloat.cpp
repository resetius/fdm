#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <random>
#include <vector>
#include <type_traits>
#include <chrono>
#include <iostream>

#include "big_float.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;

void test_strings(void** s) {
    auto f = BigFloat<2>::FromString("1");
    assert_string_equal(f.ToString().c_str(), "1");

    f = BigFloat<2>::FromString("-1");
    assert_string_equal(f.ToString().c_str(), "-1");

    f = BigFloat<2>::FromString("0.1");
    assert_string_equal(f.ToString().c_str(), "0.09999999999999999996");

    f = BigFloat<2>::FromString("-0.1");
    assert_string_equal(f.ToString().c_str(), "-0.09999999999999999996");

    f = BigFloat<2>::FromString("0.01");
    assert_string_equal(f.ToString().c_str(), "0.00999999999999999999");

    f = BigFloat<2>::FromString("0.0000123");
    assert_string_equal(f.ToString().c_str(), "0.00001229999999999997");

    f = BigFloat<2>::FromString("2.0000123");
    assert_string_equal(f.ToString().c_str(), "2.00001229999999999981");

    f = BigFloat<2>::FromString("-2.0000123");
    assert_string_equal(f.ToString().c_str(), "-2.00001229999999999981");
}

void test_sum(void** s) {
    auto a = BigFloat<64>::FromString("0.2");
    auto b = BigFloat<64>::FromString("0.3");
    auto c = a+b;
    assert_string_equal(c.ToString().c_str(), "0.49999999999999999999");

    a = BigFloat<64>::FromString("1.2");
    b = BigFloat<64>::FromString("4.3");
    c = a+b;
    assert_string_equal(c.ToString().c_str(), "5.49999999999999999999");


    a = BigFloat<64>::FromString("-1.2");
    b = BigFloat<64>::FromString("4.3");
    c = b+a;
    assert_string_equal(c.ToString().c_str(), "3.1");

    //auto f = BigFloat<64>::FromString("0.000000012");
    //std::cerr << f.ToString() << "\n";

    //f = BigFloat<64>::FromString("0.00000001");
    //std::cerr << f.ToString() << "\n";

    //f = BigFloat<64>::FromString("0.3");
    //std::cerr << f.ToString() << "\n";

    //f = BigFloat<64>::FromString("0.000000012");
    //std::cerr << f.ToString() << "\n";

    //auto f = BigFloat<64>::FromString("1.01");
    //std::cerr << "1:" << f.ToString() << "\n";

    //c = a*b;
    //std::cerr << c.ToString() << "\n";
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_strings),
        cmocka_unit_test(test_sum)
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
