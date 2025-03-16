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


    a = BigFloat<4,T>::FromString("100.2");
    b = BigFloat<4,T>::FromString("4.3");
    c = a+b;
    if constexpr(std::is_same_v<T,uint32_t>) {
        assert_string_equal(c.ToString().c_str(), "104.499999999999999993");
    } else {
        assert_string_equal(c.ToString().c_str(), "104.499999999999999999");
    }

    c = b+a;
    if constexpr(std::is_same_v<T,uint32_t>) {
        assert_string_equal(c.ToString().c_str(), "104.499999999999999993");
    } else {
        assert_string_equal(c.ToString().c_str(), "104.499999999999999999");
    }

    c = a-b;
    if constexpr(std::is_same_v<T,uint32_t>) {
        assert_string_equal(c.ToString().c_str(), "95.900000000000000001");
    } else {
        assert_string_equal(c.ToString().c_str(), "95.900000000000000000");
    }

    c = b-a;
    if constexpr(std::is_same_v<T,uint32_t>) {
        assert_string_equal(c.ToString().c_str(), "-95.900000000000000001");
    } else {
        assert_string_equal(c.ToString().c_str(), "-95.900000000000000000");
    }
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
void test_neg(void** s) {
    auto a = BigFloat<2,T>::FromString("0.5");
    assert_string_equal(a.ToString().c_str(), "0.5");
    a = -a;
    assert_string_equal(a.ToString().c_str(), "-0.5");
}

/*
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
*/

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
    auto a = BigFloat<2,T>(0.2);
    assert_string_equal(a.ToString().c_str(), "0.200000000000000011");

    a = BigFloat<2,T>(2);
    assert_string_equal(a.ToString().c_str(), "2");

    a = BigFloat<2,T>(-2);
    assert_string_equal(a.ToString().c_str(), "-2");

    a = BigFloat<2,T>(2.01);
    assert_string_equal(a.ToString().c_str(), "2.009999999999999786");
}

template<typename T>
void test_to_double(void** s) {
    auto a = BigFloat<2,T>(0.2);
    assert_double_equal(a.ToDouble(), 0.2, 1e-15);

    a = BigFloat<2,T>(2.0);
    assert_double_equal(a.ToDouble(), 2, 1e-15);

    a = BigFloat<2,T>(-2L);
    assert_double_equal(a.ToDouble(), -2, 1e-15);

    a = BigFloat<2,T>(2.01);
    assert_double_equal(a.ToDouble(), 2.01, 1e-15);
}

template<typename T>
void test_comparison(void** s) {
    auto a = BigFloat<2,T>(0.2);
    auto b = BigFloat<2,T>(0.3);
    assert_true(a < b);
    assert_true(b > a);
    assert_false(a > b);
    assert_false(b < a);

    a = BigFloat<2,T>(0.2);
    b = BigFloat<2,T>(0.2);
    assert_false(a < b);
    assert_false(b > a);
    assert_false(a > b);
    assert_false(b < a);

    a = BigFloat<2,T>(-0.2);
    b = BigFloat<2,T>(0.2);
    assert_true(a < b);
    assert_true(b > a);
    assert_false(a > b);
    assert_false(b < a);

    a = BigFloat<2,T>(-0.2);
    b = BigFloat<2,T>(-0.3);
    assert_false(a < b);
    assert_false(b > a);
    assert_true(a > b);
    assert_true(b < a);

    a = BigFloat<2,T>(2);
    auto& a_mantissa = const_cast<std::array<T,2>&>(a.getMantissa());
    a_mantissa[1] = 2;
    a_mantissa[0] = 1;

    b = BigFloat<2,T>(3);
    auto& b_mantissa = const_cast<std::array<T,2>&>(b.getMantissa());
    b_mantissa[1] = 1;
    b_mantissa[0] = 9;
    assert_true(a > b);
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
        auto one = BigFloat<2,T>(1.0);
        auto eps = BigFloat<2,T>(1.0);

        int i = 0;
        while (one + eps > one) {
            eps = BigFloat<2,T>(0.5) * eps;
            i++;
        }

        assert_true(i == 64 || i == 128);
    }

    {
        auto one = BigFloat<4,T>(1.0);
        auto eps = BigFloat<4,T>(1.0);

        int i = 0;
        while (one + eps > one) {
            eps = BigFloat<4,T>(0.5) * eps;
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
void bench_dif(const std::string& name)
{
    T a = 0.2;
    T b = 0.3;
    T c = 0.0;

    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000000; i++) {
        c = c - a - b;
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
void bench_square(const std::string& name)
{
    T b = 0.3;
    T c = 0.0;

    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 1000000; i++) {
        c = c + b * b;
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
    //bench_sum<_Float128>("sum float128");

    bench_dif<BigFloat<2,T>>("dif BigFloat<2>");
    bench_dif<BigFloat<4,T>>("dif BigFloat<4>");
    bench_dif<BigFloat<8,T>>("dif BigFloat<8>");
    bench_dif<BigFloat<16,T>>("dif BigFloat<16>");
    bench_dif<float>("dif float");
    bench_dif<double>("dif double");

    bench_mul<BigFloat<2,T>>("mul BigFloat<2>");
    bench_mul<BigFloat<4,T>>("mul BigFloat<4>");
    bench_mul<BigFloat<8,T>>("mul BigFloat<8>");
    bench_mul<BigFloat<16,T>>("mul BigFloat<16>");
    bench_mul<float>("mul float");
    bench_mul<double>("mul double");
    //bench_mul<_Float128>("mul float128");

    bench_square<BigFloat<2,T>>("sq BigFloat<2>");
    bench_square<BigFloat<4,T>>("sq BigFloat<4>");
    bench_square<BigFloat<8,T>>("sq BigFloat<8>");
    bench_square<BigFloat<16,T>>("sq BigFloat<16>");
    bench_square<float>("sq float");
    bench_square<double>("sq double");
    //bench_square<_Float128>("sq float128");
}

template<typename T>
void test_construct(void**) {
    BigFloat<2,T> v1(123.456);
    BigFloat<4,T> v2(v1);
    assert_double_equal((double)v2, (double)v1, 1e-15);

    BigFloat<3,T> v3(123.456);
    BigFloat<2,T> v4(v3);
    assert_double_equal((double)v3, (double)v4, 1e-15);
}

void test_precision(void**)
{
    // AdditionPrecision
    {
        BigFloat<2,uint32_t> a, b;
        a.sign = b.sign = 1;
        a.mantissa[1] = 1; // 2^32
        a.exponent = 0;
        a.normalize();

        b.mantissa[0] = 1;
        b.exponent = -31;
        b.normalize();

        BigFloat<2,uint32_t> sum = a + b;
        assert_true(sum.mantissa[1] == 1U<<31 && sum.mantissa[0] == 1);
    }
    // BorrowHandling
    {
        BigFloat<2,uint32_t> a, b;
        a.mantissa[1] = 1; // 2^32
        a.exponent = 0;
        a.normalize();

        b.mantissa[0] = 0xFFFFFFFF;
        b.exponent = 0;
        b.normalize();

        BigFloat<2,uint32_t> diff = a - b;

        assert_true(diff.mantissa[1] == 1U<<31 && diff.mantissa[0] == 0);
    }
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
        my_unit(test_neg),
        //my_unit(test_div),
        my_unit(test_long),
        my_unit(test_from_double),
        my_unit(test_to_double),
        my_unit(test_comparison),
        my_unit(test_mandelbrot),
        my_unit(test_eps),
        my_unit(test_microbench),
        my_unit(test_construct),
        cmocka_unit_test(test_precision),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
