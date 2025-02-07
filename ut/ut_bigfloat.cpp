#include "big_float.h"

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
    auto a = BigFloat<2>::FromString("0.2");
    auto b = BigFloat<2>::FromString("0.3");
    auto c = a+b;
    assert_string_equal(c.ToString().c_str(), "0.49999999999999999994");

    a = BigFloat<2>::FromString("1.2");
    b = BigFloat<2>::FromString("4.3");
    c = a+b;
    assert_string_equal(c.ToString().c_str(), "5.49999999999999999956");

    a = BigFloat<2>::FromString("-1.2");
    b = BigFloat<2>::FromString("4.3");
    c = b+a;
    assert_string_equal(c.ToString().c_str(), "3.09999999999999999991");

    a = BigFloat<2>::FromString("1.2");
    b = BigFloat<2>::FromString("4.3");
    c = b-a;
    assert_string_equal(c.ToString().c_str(), "3.09999999999999999991");

    a = BigFloat<2>::FromString("1.2");
    b = BigFloat<2>::FromString("4.3");
    c = a-b;
    assert_string_equal(c.ToString().c_str(), "-3.09999999999999999991");

}

void test_mul(void** s) {
    auto a = BigFloat<2>::FromString("0.2");
    auto b = BigFloat<2>::FromString("0.3");
    auto c = a*b;
    assert_string_equal(c.ToString().c_str(), "0.05999999999999999998");

    a = BigFloat<2>::FromString("-0.2");
    b = BigFloat<2>::FromString("0.3");
    c = a*b;
    assert_string_equal(c.ToString().c_str(), "-0.05999999999999999998");

    a = BigFloat<2>::FromString("0.5");
    b = BigFloat<2>::FromString("1.1");
    c = a*b;
    assert_string_equal(c.ToString().c_str(), "0.54999999999999999995");
}

void test_long(void** s) {
    auto a = BigFloat<128>::FromString("0.2");
    auto b = BigFloat<128>::FromString("0.3");
    auto c = a*b;
    assert_string_equal(c.ToString().c_str(), "0.05999999999999999999");

    a = BigFloat<128>::FromString("0.0000002");
    b = BigFloat<128>::FromString("0.3");
    c = a*b;
    assert_string_equal(c.ToString().c_str(), "0.00000005999999999999");
}

void test_from_double(void** s) {
    auto a = BigFloat<2>::FromDouble(0.2);
    assert_string_equal(a.ToString().c_str(), "0.20000000000000001110");

    a = BigFloat<2>::FromDouble(2);
    assert_string_equal(a.ToString().c_str(), "2");

    a = BigFloat<2>::FromDouble(-2);
    assert_string_equal(a.ToString().c_str(), "-2");

    a = BigFloat<2>::FromDouble(2.01);
    assert_string_equal(a.ToString().c_str(), "2.00999999999999978683");
}

void test_to_double(void** s) {
    auto a = BigFloat<2>::FromDouble(0.2);
    assert_double_equal(a.ToDouble(), 0.2, 1e-15);

    a = BigFloat<2>::FromDouble(2);
    assert_double_equal(a.ToDouble(), 2, 1e-15);

    a = BigFloat<2>::FromDouble(-2);
    assert_double_equal(a.ToDouble(), -2, 1e-15);

    a = BigFloat<2>::FromDouble(2.01);
    assert_double_equal(a.ToDouble(), 2.01, 1e-15);
}

void test_comparison(void** s) {
    auto a = BigFloat<2>::FromDouble(0.2);
    auto b = BigFloat<2>::FromDouble(0.3);
    assert_true(a < b);
    assert_true(b > a);
    assert_false(a > b);
    assert_false(b < a);

    a = BigFloat<2>::FromDouble(0.2);
    b = BigFloat<2>::FromDouble(0.2);
    assert_false(a < b);
    assert_false(b > a);
    assert_false(a > b);
    assert_false(b < a);

    a = BigFloat<2>::FromDouble(-0.2);
    b = BigFloat<2>::FromDouble(0.2);
    assert_true(a < b);
    assert_true(b > a);
    assert_false(a > b);
    assert_false(b < a);

    a = BigFloat<2>::FromDouble(-0.2);
    b = BigFloat<2>::FromDouble(-0.3);
    assert_false(a < b);
    assert_false(b > a);
    assert_true(a > b);
    assert_true(b < a);
}

template<typename T>
int mandelbrot(T ca, T cb) {
    T za = 0.0;
    T zb = 0.0;
    for (int i = 0; i < 1000; i++) {
        if ((za*za + zb*zb) > T(4.0)) {
            return i;
        }
        T za_new = za*za - zb*zb + ca;
        zb = T(2.0)*za*zb + cb;
        za = za_new;
    }
    return 1000;
}

void test_mandelbrot(void** s) {
    int iters1 = 0, iters2 = 0;
    {
        using T = BigFloat<4>;
        iters1 = mandelbrot<T>(-1.2, 0.0);
    }
    {
        using T = double;
        iters2 = mandelbrot<T>(-1.2, 0.0);
    }

    assert_int_equal(iters1, iters2);
}

void test_eps(void** s) {
    {
        auto one = BigFloat<2>::FromDouble(1.0);
        auto eps = BigFloat<2>::FromDouble(1.0);

        int i = 0;
        while (one + eps > one) {
            eps = BigFloat<2>::FromDouble(0.5) * eps;
            i++;
        }

        assert_int_equal(i, 64);
    }

    {
        auto one = BigFloat<4>::FromDouble(1.0);
        auto eps = BigFloat<4>::FromDouble(1.0);

        int i = 0;
        while (one + eps > one) {
            eps = BigFloat<4>::FromDouble(0.5) * eps;
            i++;
        }

        assert_int_equal(i, 128);
    }
}

template<typename T>
void bench_sum(const std::string& name)
{
    T a = 0.2;
    T b = 0.3;
    T c = 0.0;

    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000000; i++) {
        c = c + a + b;
    }
    auto t2 = std::chrono::steady_clock::now();
    auto interval = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cerr << name << " : " << interval.count() << std::endl;
}

template<typename T>
void bench_mul(const std::string& name)
{
    T a = 0.2;
    T b = 0.3;
    T c = 0.0;

    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000000; i++) {
        c = c + a * b;
    }
    auto t2 = std::chrono::steady_clock::now();
    auto interval = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cerr << name << " : " << interval.count() << std::endl;
}

void test_microbench(void**) {
    bench_sum<BigFloat<2>>("sum BigFloat<2>");
    bench_sum<BigFloat<4>>("sum BigFloat<4>");
    bench_sum<BigFloat<8>>("sum BigFloat<8>");
    bench_sum<BigFloat<16>>("sum BigFloat<16>");
    bench_sum<float>("sum float");
    bench_sum<double>("sum double");

    bench_mul<BigFloat<2>>("mul BigFloat<2>");
    bench_mul<BigFloat<4>>("mul BigFloat<4>");
    bench_mul<BigFloat<8>>("mul BigFloat<8>");
    bench_mul<BigFloat<16>>("mul BigFloat<16>");
    bench_mul<float>("mul float");
    bench_mul<double>("mul double");
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_strings),
        cmocka_unit_test(test_sum),
        cmocka_unit_test(test_mul),
        cmocka_unit_test(test_long),
        cmocka_unit_test(test_from_double),
        cmocka_unit_test(test_to_double),
        cmocka_unit_test(test_comparison),
        cmocka_unit_test(test_mandelbrot),
        cmocka_unit_test(test_eps),
        cmocka_unit_test(test_microbench),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
