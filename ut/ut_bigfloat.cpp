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

template<typename T>
void test_strings(void** s) {
    auto f = BigFloat<2,T>::FromString("1");
    assert_string_equal(f.ToString().c_str(), "1");

    f = BigFloat<2,T>::FromString("-1");
    assert_string_equal(f.ToString().c_str(), "-1");

    f = BigFloat<2,T>::FromString("0.1");
    assert_string_equal(f.ToString().c_str(), "0.099999999999999999");

    f = BigFloat<2,T>::FromString("-0.1");
    assert_string_equal(f.ToString().c_str(), "-0.099999999999999999");

    f = BigFloat<2,T>::FromString("0.01");
    assert_string_equal(f.ToString().c_str(), "0.009999999999999999");

    f = BigFloat<2,T>::FromString("0.0000123");
    assert_string_equal(f.ToString().c_str(), "0.000012299999999999");

    f = BigFloat<2,T>::FromString("2.0000123");
    assert_string_equal(f.ToString().c_str(), "2.000012299999999999");

    f = BigFloat<2,T>::FromString("-2.0000123");
    assert_string_equal(f.ToString().c_str(), "-2.000012299999999999");
}

template<typename T>
void test_sum(void** s) {
    auto a = BigFloat<2,T>::FromString("0.2");
    auto b = BigFloat<2,T>::FromString("0.3");
    auto c = a+b;
    assert_string_equal(c.ToString().c_str(), "0.499999999999999999");

    a = BigFloat<2,T>::FromString("1.2");
    b = BigFloat<2,T>::FromString("4.3");
    c = a+b;
    assert_string_equal(c.ToString().c_str(), "5.499999999999999999");

    a = BigFloat<2,T>::FromString("-1.2");
    b = BigFloat<2,T>::FromString("4.3");
    c = b+a;
    assert_string_equal(c.ToString().c_str(), "3.099999999999999999");

    a = BigFloat<2,T>::FromString("1.2");
    b = BigFloat<2,T>::FromString("4.3");
    c = b-a;
    assert_string_equal(c.ToString().c_str(), "3.099999999999999999");

    a = BigFloat<2,T>::FromString("1.2");
    b = BigFloat<2,T>::FromString("4.3");
    c = a-b;
    assert_string_equal(c.ToString().c_str(), "-3.099999999999999999");

}

template<typename T>
void test_mul(void** s) {
    auto a = BigFloat<2,T>::FromString("0.5");
    auto b = BigFloat<2,T>::FromString("0.5");
    auto c = a*b;
    assert_string_equal(c.ToString().c_str(), "0.25");

    a = BigFloat<2,T>::FromString("0.2");
    b = BigFloat<2,T>::FromString("0.3");
    c = a*b;
    assert_string_equal(c.ToString().c_str(), "0.059999999999999999");

    a = BigFloat<2,T>::FromString("-0.2");
    b = BigFloat<2,T>::FromString("0.3");
    c = a*b;
    assert_string_equal(c.ToString().c_str(), "-0.059999999999999999");

    a = BigFloat<2,T>::FromString("0.5");
    b = BigFloat<2,T>::FromString("1.1");
    c = a*b;
    assert_string_equal(c.ToString().c_str(), "0.549999999999999999");
}

template<typename T>
void test_div(void** s) {
    auto a = BigFloat<2,T>::FromString("0.5");
    auto b = BigFloat<2,T>::FromString("2");
    assert_string_equal(b.Inv().ToString().c_str(), "0.5");

    b = BigFloat<2,T>::FromString("4");
    assert_string_equal(b.Inv().ToString().c_str(), "0.25");

    b = BigFloat<2,T>::FromString("8");
    assert_string_equal(b.Inv().ToString().c_str(), "0.125");

    auto c = a/b;
    assert_string_equal(c.ToString().c_str(), "0.0625");

    a = BigFloat<2,T>::FromString("0.2");
    b = BigFloat<2,T>::FromString("0.3");
    c = a/b;
    assert_string_equal(c.ToString().c_str(), "0.059999999999999999");

    a = BigFloat<2,T>::FromString("-0.2");
    b = BigFloat<2,T>::FromString("0.3");
    c = a/b;
    assert_string_equal(c.ToString().c_str(), "-0.059999999999999999");

    a = BigFloat<2,T>::FromString("0.5");
    b = BigFloat<2,T>::FromString("1.1");
    c = a/b;
    assert_string_equal(c.ToString().c_str(), "0.549999999999999999");
}

template<typename T>
void test_long(void** s) {
    auto a = BigFloat<128,T>::FromString("0.2");
    auto b = BigFloat<128,T>::FromString("0.3");
    auto c = a*b;
    assert_string_equal(c.ToString().c_str(), "0.059999999999999999");

    a = BigFloat<128,T>::FromString("0.0000002");
    b = BigFloat<128,T>::FromString("0.3");
    c = a*b;
    assert_string_equal(c.ToString().c_str(), "0.000000059999999999");
}

template<typename T>
void test_from_double(void** s) {
    auto a = BigFloat<2,T>::FromDouble(0.2);
    assert_string_equal(a.ToString().c_str(), "0.200000000000000011");

    a = BigFloat<2,T>::FromDouble(2);
    assert_string_equal(a.ToString().c_str(), "2");

    a = BigFloat<2,T>::FromDouble(-2);
    assert_string_equal(a.ToString().c_str(), "-2");

    a = BigFloat<2,T>::FromDouble(2.01);
    assert_string_equal(a.ToString().c_str(), "2.009999999999999786");
}

template<typename T>
void test_to_double(void** s) {
    auto a = BigFloat<2,T>::FromDouble(0.2);
    assert_double_equal(a.ToDouble(), 0.2, 1e-15);

    a = BigFloat<2,T>::FromDouble(2);
    assert_double_equal(a.ToDouble(), 2, 1e-15);

    a = BigFloat<2,T>::FromDouble(-2);
    assert_double_equal(a.ToDouble(), -2, 1e-15);

    a = BigFloat<2,T>::FromDouble(2.01);
    assert_double_equal(a.ToDouble(), 2.01, 1e-15);
}

template<typename T>
void test_comparison(void** s) {
    auto a = BigFloat<2,T>::FromDouble(0.2);
    auto b = BigFloat<2,T>::FromDouble(0.3);
    assert_true(a < b);
    assert_true(b > a);
    assert_false(a > b);
    assert_false(b < a);

    a = BigFloat<2,T>::FromDouble(0.2);
    b = BigFloat<2,T>::FromDouble(0.2);
    assert_false(a < b);
    assert_false(b > a);
    assert_false(a > b);
    assert_false(b < a);

    a = BigFloat<2,T>::FromDouble(-0.2);
    b = BigFloat<2,T>::FromDouble(0.2);
    assert_true(a < b);
    assert_true(b > a);
    assert_false(a > b);
    assert_false(b < a);

    a = BigFloat<2,T>::FromDouble(-0.2);
    b = BigFloat<2,T>::FromDouble(-0.3);
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

template<typename T>
void test_mandelbrot(void** s) {
    int iters1 = 0, iters2 = 0;
    {
        using U = BigFloat<4,T>;
        iters1 = mandelbrot<U>(-1.2, 0.0);
    }
    {
        using U = double;
        iters2 = mandelbrot<U>(-1.2, 0.0);
    }

    assert_int_equal(iters1, iters2);
}

template<typename T>
void test_eps(void** s) {
    {
        auto one = BigFloat<2,T>::FromDouble(1.0);
        auto eps = BigFloat<2,T>::FromDouble(1.0);

        int i = 0;
        while (one + eps > one) {
            eps = BigFloat<2,T>::FromDouble(0.5) * eps;
            i++;
        }

        assert_true(i == 64 || i == 128);
    }

    {
        auto one = BigFloat<4,T>::FromDouble(1.0);
        auto eps = BigFloat<4,T>::FromDouble(1.0);

        int i = 0;
        while (one + eps > one) {
            eps = BigFloat<4,T>::FromDouble(0.5) * eps;
            i++;
        }

        assert_true(i == 128 || i == 256);
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

template<typename T>
void test_microbench(void**) {
    bench_sum<BigFloat<2,T>>("sum BigFloat<2>");
    bench_sum<BigFloat<4,T>>("sum BigFloat<4>");
    bench_sum<BigFloat<8,T>>("sum BigFloat<8>");
    bench_sum<BigFloat<16,T>>("sum BigFloat<16>");
    bench_sum<float>("sum float");
    bench_sum<double>("sum double");

    bench_mul<BigFloat<2,T>>("mul BigFloat<2>");
    bench_mul<BigFloat<4,T>>("mul BigFloat<4>");
    bench_mul<BigFloat<8,T>>("mul BigFloat<8>");
    bench_mul<BigFloat<16,T>>("mul BigFloat<16>");
    bench_mul<float>("mul float");
    bench_mul<double>("mul double");
}

#define my_unit_test2(f, a, b) \
    { #f "(" #a ")", f<a>, NULL, NULL, NULL }, \
    { #f "(" #b ")", f<b>, NULL, NULL, NULL }

#define my_unit(f) my_unit_test2(f, uint32_t, uint64_t)

int main() {
    const struct CMUnitTest tests[] = {
        my_unit(test_strings),
        my_unit(test_sum),
        my_unit(test_mul),
        //my_unit(test_div),
        my_unit(test_long),
        my_unit(test_from_double),
        my_unit(test_to_double),
        my_unit(test_comparison),
        my_unit(test_mandelbrot),
        my_unit(test_eps),
        my_unit(test_microbench),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
