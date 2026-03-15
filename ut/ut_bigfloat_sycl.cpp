// acpp --acpp-targets=<target> -std=c++20 ut_bigfloat_sycl.cpp -I../src -lcmocka -o ut_bigfloat_sycl
#include "big_float.h"

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <vector>
#include <type_traits>
#include <iostream>

#include <sycl/sycl.hpp>

extern "C" {
#include <cmocka.h>
}

static sycl::queue& get_queue() {
    static sycl::queue q{sycl::default_selector_v};
    return q;
}

// Mandelbrot using integer literals - no double inside kernel body
template<typename T>
int mandelbrot_sycl_kernel(T ca, T cb) {
    T za(0);
    T zb(0);
    T four(4);
    for (int i = 0; i < 1000; i++) {
        if ((za*za + zb*zb) > four) {
            return i;
        }
        T za_new = za*za - zb*zb + ca;
        zb = (za*zb).Mul2() + cb;
        za = za_new;
    }
    return 1000;
}

static int mandelbrot_double(double ca, double cb) {
    double za = 0.0, zb = 0.0;
    for (int i = 0; i < 1000; i++) {
        if (za*za + zb*zb > 4.0) return i;
        double za_new = za*za - zb*zb + ca;
        zb = 2.0*za*zb + cb;
        za = za_new;
    }
    return 1000;
}

template<typename T, typename Spec>
void test_sycl_sum(void** s) {
    sycl::queue& q = get_queue();

    // All inputs built on host, kernel does only integer arithmetic
    struct Results {
        BigFloat<2,T,Spec> r[8];
        BigFloat<4,T,Spec> r4[4];
    };

    auto a2 = BigFloat<2,T,Spec>::FromString("0.2");
    auto b3 = BigFloat<2,T,Spec>::FromString("0.3");
    auto a12 = BigFloat<2,T,Spec>::FromString("1.2");
    auto b43 = BigFloat<2,T,Spec>::FromString("4.3");
    auto an12 = BigFloat<2,T,Spec>::FromString("-1.2");
    auto a100 = BigFloat<4,T,Spec>::FromString("100.2");
    auto b4 = BigFloat<4,T,Spec>::FromString("4.3");

    Results* res = sycl::malloc_shared<Results>(1, q);

    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            res->r[0] = a2 + b3;
            res->r[1] = a12 + b43;
            res->r[2] = b43 + an12;
            res->r[3] = b43 - a12;
            res->r[4] = a12 - b43;
            res->r4[0] = a100 + b4;
            res->r4[1] = b4 + a100;
            res->r4[2] = a100 - b4;
            res->r4[3] = b4 - a100;
        });
    }).wait();

    assert_string_equal(res->r[0].ToString().c_str(), "0.499999999999999999");
    assert_string_equal(res->r[1].ToString().c_str(), "5.499999999999999999");
    assert_string_equal(res->r[2].ToString().c_str(), "3.099999999999999999");
    assert_string_equal(res->r[3].ToString().c_str(), "3.099999999999999999");
    assert_string_equal(res->r[4].ToString().c_str(), "-3.099999999999999999");

    // GenericPlatformSpec gives the same result for uint32_t and uint64_t
    // (unlike AMD64PlatformSpec<uint32_t> which gives "..993"/"..001")
    assert_string_equal(res->r4[0].ToString().c_str(), "104.499999999999999999");
    assert_string_equal(res->r4[1].ToString().c_str(), "104.499999999999999999");
    assert_string_equal(res->r4[2].ToString().c_str(), "95.900000000000000000");
    assert_string_equal(res->r4[3].ToString().c_str(), "-95.900000000000000000");

    sycl::free(res, q);
}

template<typename T, typename Spec>
void test_sycl_mul(void** s) {
    sycl::queue& q = get_queue();

    struct Results {
        BigFloat<2,T,Spec> r[4];
    };

    auto a05 = BigFloat<2,T,Spec>::FromString("0.5");
    auto b05 = BigFloat<2,T,Spec>::FromString("0.5");
    auto a02 = BigFloat<2,T,Spec>::FromString("0.2");
    auto b03 = BigFloat<2,T,Spec>::FromString("0.3");
    auto an02 = BigFloat<2,T,Spec>::FromString("-0.2");
    auto b11 = BigFloat<2,T,Spec>::FromString("1.1");

    Results* res = sycl::malloc_shared<Results>(1, q);

    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            res->r[0] = a05 * b05;
            res->r[1] = a02 * b03;
            res->r[2] = an02 * b03;
            res->r[3] = a05 * b11;
        });
    }).wait();

    assert_string_equal(res->r[0].ToString().c_str(), "0.25");
    assert_string_equal(res->r[1].ToString().c_str(), "0.059999999999999999");
    assert_string_equal(res->r[2].ToString().c_str(), "-0.059999999999999999");
    assert_string_equal(res->r[3].ToString().c_str(), "0.549999999999999999");

    sycl::free(res, q);
}

template<typename T, typename Spec>
void test_sycl_neg(void** s) {
    sycl::queue& q = get_queue();

    auto a = BigFloat<2,T,Spec>::FromString("0.5");
    BigFloat<2,T,Spec>* res = sycl::malloc_shared<BigFloat<2,T,Spec>>(1, q);

    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            *res = -a;
        });
    }).wait();

    assert_string_equal(res->ToString().c_str(), "-0.5");

    sycl::free(res, q);
}

// Mandelbrot: ca/cb constructed on host from double, kernel uses integer literals only
template<typename T, typename Spec>
void test_sycl_mandelbrot(void** s) {
    sycl::queue& q = get_queue();

    using U = BigFloat<4,T,Spec>;

    // Inputs built on host - double constructor runs on host, not inside kernel
    U ca(-1.2);
    U cb(0.0);

    int* res = sycl::malloc_shared<int>(1, q);

    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            // mandelbrot_sycl_kernel uses only integer literals inside - no fp64 needed
            *res = mandelbrot_sycl_kernel(ca, cb);
        });
    }).wait();

    int expected = mandelbrot_double(-1.2, 0.0);
    assert_int_equal(*res, expected);

    sycl::free(res, q);
}

// Tests that use double inside the kernel - gated by fp64 support
template<typename T, typename Spec>
void test_sycl_from_double(void** s) {
    sycl::queue& q = get_queue();

    if (!q.get_device().has(sycl::aspect::fp64)) {
        std::cerr << "[  SKIP  ] fp64 not supported on "
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";
        skip();
        return;
    }

    struct Results {
        BigFloat<2,T,Spec> r[4];
    };
    Results* res = sycl::malloc_shared<Results>(1, q);

    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            res->r[0] = BigFloat<2,T,Spec>(0.2);
            res->r[1] = BigFloat<2,T,Spec>(2.0);
            res->r[2] = BigFloat<2,T,Spec>(-2.0);
            res->r[3] = BigFloat<2,T,Spec>(2.01);
        });
    }).wait();

    assert_string_equal(res->r[0].ToString().c_str(), "0.200000000000000011");
    assert_string_equal(res->r[1].ToString().c_str(), "2");
    assert_string_equal(res->r[2].ToString().c_str(), "-2");
    assert_string_equal(res->r[3].ToString().c_str(), "2.009999999999999786");

    sycl::free(res, q);
}

template<typename T, typename Spec>
void test_sycl_to_double(void** s) {
    sycl::queue& q = get_queue();

    if (!q.get_device().has(sycl::aspect::fp64)) {
        std::cerr << "[  SKIP  ] fp64 not supported on "
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";
        skip();
        return;
    }

    auto a02 = BigFloat<2,T,Spec>(0.2);
    auto a20 = BigFloat<2,T,Spec>(2.0);
    auto an2 = BigFloat<2,T,Spec>(-2L);
    auto a201 = BigFloat<2,T,Spec>(2.01);

    double* res = sycl::malloc_shared<double>(4, q);

    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            res[0] = a02.ToDouble();
            res[1] = a20.ToDouble();
            res[2] = an2.ToDouble();
            res[3] = a201.ToDouble();
        });
    }).wait();

    assert_double_equal(res[0], 0.2,  1e-15);
    assert_double_equal(res[1], 2.0,  1e-15);
    assert_double_equal(res[2], -2.0, 1e-15);
    assert_double_equal(res[3], 2.01, 1e-15);

    sycl::free(res, q);
}

// GenericPlatformSpec only: NaivePlatformSpec has __uint128_t in a dead
// if constexpr branch that icpx NVPTX backend still rejects.
// GenericPlatformSpec<uint64_t> is safe: no __int128 in device-callable
// paths (WideType/__int128 only appears in ToString which is host-only).
#define my_unit(f) \
    { #f "(uint32_t,generic)", f<uint32_t,GenericPlatformSpec<uint32_t>>, NULL, NULL, NULL }, \
    { #f "(uint64_t,generic)", f<uint64_t,GenericPlatformSpec<uint64_t>>, NULL, NULL, NULL }

int main() {
    auto& q = get_queue();
    std::cerr << "SYCL device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cerr << "fp64 support: "
              << (q.get_device().has(sycl::aspect::fp64) ? "yes" : "no") << "\n";

    const struct CMUnitTest tests[] = {
        my_unit(test_sycl_sum),
        my_unit(test_sycl_mul),
        my_unit(test_sycl_neg),
        my_unit(test_sycl_mandelbrot),
        my_unit(test_sycl_from_double),
        my_unit(test_sycl_to_double),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
