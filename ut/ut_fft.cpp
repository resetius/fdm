#include <memory>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <random>
#include <vector>
#include <type_traits>
#include <chrono>

#include "asp_fft.h"
#include "fft.h"
#include "config.h"

#ifdef HAVE_ONEMATH
#include "fft_sycl.h"
#endif

extern "C" {
#include <cmocka.h>
}

using namespace std;
using namespace std::chrono;
using namespace fdm;

void test_periodic(void**) {
    using T = double;
    int N = 1024;
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<double> S(N);
    vector<double> s(N);
    vector<double> s1(N);
    fft* ft = FFT_init(FFT_PERIODIC, N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    pFFT_2(&S[0], &s[0], 1.0, ft);
    pFFT_2_1(&s1[0], &S[0], 2.0/N, ft);
    for (int i = 0; i < N; i++) {
        assert_float_equal(s1[i], s[i], 1e-15);
    }

    FFT_free(ft);
}

template<typename T>
void test_periodic_new(void**) {
    int N = 1024;
    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-5;
    }
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s; // s1 will be utilized by pFFT
    ft.pFFT(&S[0], &s1[0], 1.0);
    ft.pFFT_1(&s1[0], &S[0], 2.0/N);
    for (int i = 0; i < N; i++) {
        assert_float_equal(s1[i], s[i], tol);
    }
}

void test_periodic_new_double(void**) {
    test_periodic_new<double>(nullptr);
}

void test_periodic_new_float(void**) {
    test_periodic_new<float>(nullptr);
}

template<typename T>
void test_periodic_3d(void**) {
    int N = 128;
    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-5;
    }
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<T> S(N*N*N);
    vector<T> s(N*N*N);
    vector<T> s1(N*N*N);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);

    for (int i = 0; i < N*N*N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s; // s1 will be utilized by pFFT
    ft.pFFT_3d(&S[0], &s1[0], 1.0);
    ft.pFFT_1_3d(&s1[0], &S[0], 2.0/N);
    for (int i = 0; i < N*N*N; i++) {
        assert_float_equal(s1[i], s[i], tol);
    }
}

void test_periodic_3d_double(void**) {
    test_periodic_3d<double>(nullptr);
}

void test_periodic_3d_float(void**) {
    test_periodic_3d<float>(nullptr);
}

#ifdef HAVE_ONEMATH
template<typename T>
void test_periodic_3d_sycl(void**) {
    sycl::queue q{ sycl::default_selector_v };
    int N = 8;
    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-5;
    }
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    using alloc_type = sycl_allocator<T>;
    alloc_type alloc(q);
    vector<T, alloc_type> S(N*N*N, alloc);
    vector<T, alloc_type> S1(N*N*N, alloc);
    vector<T, alloc_type> S2(N*N*N, alloc);
    vector<T, alloc_type> s(N*N*N, alloc);
    vector<T, alloc_type> s1(N*N*N, alloc);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);

    fdm::FFTSycl<T> fft_sycl(q, N);

    for (int i = 0; i < N*N*N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s; // s1 will be utilized by pFFT
    fft_sycl.pFFT_3d(&S[0], &s1[0], 1.0).wait();
    S2 = S;

    s1 = s;
    ft.pFFT_3d(&S1[0], &s1[0], 1.0);
    for (int i = 0; i < N*N*N; i++) {
        assert_float_equal(S2[i], S1[i], tol);
    }

    s1 = s;
    fft_sycl.pFFT_1_3d(&S[0], &s1[0], 1.0).wait();
    S2 = S;

    s1 = s;
    ft.pFFT_1_3d(&S1[0], &s1[0], 1.0);
    for (int i = 0; i < N*N*N; i++) {
        assert_float_equal(S2[i], S1[i], tol);
    }

    s1 = s;
    fft_sycl.pFFT_3d(&S[0], &s1[0], 1.0).wait();
    fft_sycl.pFFT_1_3d(&s1[0], &S[0], 2.0/N).wait();

    for (int i = 0; i < N*N*N; i++) {
        assert_float_equal(s1[i], s[i], tol);
    }
}

void test_periodic_3d_sycl_double(void**) {
    test_periodic_3d_sycl<double>(nullptr);
}

void test_periodic_3d_sycl_float(void**) {
    test_periodic_3d_sycl<float>(nullptr);
}
#endif

template<typename T>
void test_sin(void**) {
    int N = 1024;
    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-5;
    }
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    vector<T> s2(N);
    vector<T> s3(N);

    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s3 = s;
    ft.sFFT(&S[0], &s3[0], 1.0);
    s3 = S;
    ft.sFFT(&s1[0], &s3[0], 2.0/N);

    for (int i = 1; i < N-1; i++) {
        assert_float_equal(s1[i], s[i], tol);
    }

    if constexpr(is_same<T,double>::value) {
        sFT(&s2[0], &S[0], 2.0/N, N);
        for (int i = 1; i < N-1; i++) {
            assert_float_equal(s1[i], s2[i], tol);
        }
    }
}

void test_sin_double(void**) {
    test_sin<double>(nullptr);
}

void test_sin_float(void**) {
    test_sin<float>(nullptr);
}

template<typename T>
void test_cos(void**) {
    int N = 8;
    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-5;
    }
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<T> S(N+1);
    vector<T> s(N+1);
    vector<T> s1(N+1);
    vector<T> s2(N+1);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);

    for (int i = 0; i <= N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    ft.cFFT(&S[0], &s1[0], 1.0);

    if constexpr(is_same<T,double>::value) {
        cFT(&s2[0], &s[0], 1.0, N);
        for (int i = 0; i <= N; i++) {
            assert_float_equal(S[i], s2[i], tol);
        }
    }

    s2 = S;
    ft.cFFT(&s1[0], &s2[0], 2.0/N);

    for (int i = 0; i <= N; i++) {
        assert_float_equal(s1[i], s[i], tol);
    }

    if constexpr(is_same<T,double>::value) {
        cFT(&s2[0], &S[0], 2.0/N, N);
        for (int i = 0; i <= N; i++) {
            assert_float_equal(s1[i], s2[i], tol);
        }
    }
}

void test_cos_double(void**) {
    test_cos<double>(nullptr);
}

void test_cos_float(void**) {
    test_cos<float>(nullptr);
}

void test_periodic_new_old_cmp(void**) {
    using T = double;
    int N = 1024;
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);

    vector<double> S(N);
    vector<double> s(N);
    vector<double> s1(N);
    vector<double> S1(N);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fft* ft1 = FFT_init(FFT_PERIODIC, N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    ft.pFFT(&S[0], &s1[0], 1.0);
    pFFT_2(&S1[0], &s[0], 1.0, ft1);
    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], 1e-15);
    }

    FFT_free(ft1);
}

template<typename T>
void test_sin_new(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFT<T> ft_old(table, N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    vector<T> S1(N);
    vector<T> S2(N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    if constexpr(is_same<double,T>::value) {
        auto t1 = steady_clock::now();
        sFT(&S2[0], &s[0], 1.0, N);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t1=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft_old.sFFT(&S[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t2=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.sFFT(&S1[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose > 1) {
            printf("%e <> %e <> %e\n", S2[i], S1[i], S[i]);
        }
    }
}

void test_sin_new_double(void** s) {
    test_sin_new<double>(s);
}

void test_sin_new_float(void** s) {
    test_sin_new<float>(s);
}

#ifdef HAVE_FFTW3
template<typename T>
void test_sin_fftw3(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFT_fftw3<T> ft_fftw3(N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    vector<T> S1(N);
    vector<T> S2(N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    if constexpr(is_same<double,T>::value) {
        auto t1 = steady_clock::now();
        sFT(&S2[0], &s1[0], 1.0, N);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t1=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft_fftw3.sFFT(&S[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t2=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.sFFT(&S1[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose > 1) {
            printf("%e <> %e <> %e\n", S2[i], S1[i], S[i]);
        }
    }
}

void test_sin_fftw3_double(void** s) {
    test_sin_fftw3<double>(s);
}

void test_sin_fftw3_float(void** s) {
    test_sin_fftw3<float>(s);
}
#endif

template<typename T>
void test_cos_new(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFT_old<T> ft_old(table, N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N+1);
    vector<T> s(N+1);
    vector<T> s1(N+1);
    vector<T> S1(N+1);
    vector<T> S2(N+1);

    for (int i = 0; i < N+1; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    if constexpr(is_same<double,T>::value) {
        auto t1 = steady_clock::now();
        cFT(&S2[0], &s[0], 1.0, N);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t1=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft_old.cFFT(&S[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t2=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.cFFT(&S1[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i <= N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose > 1) {
            printf("%e <> %e <> %e\n", S2[i], S1[i], S[i]);
        }
    }
}

void test_cos_new_double(void** s) {
    test_cos_new<double>(s);
}

void test_cos_new_float(void** s) {
    test_cos_new<float>(s);
}

#ifdef HAVE_FFTW3
template<typename T>
void test_cos_fftw3(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFT_fftw3<T> ft_fftw3(N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N+1);
    vector<T> s(N+1);
    vector<T> s1(N+1);
    vector<T> S1(N+1);
    vector<T> S2(N+1);

    for (int i = 0; i < N+1; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    if constexpr(is_same<double,T>::value) {
        auto t1 = steady_clock::now();
        cFT(&S2[0], &s[0], 1.0, N);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t1=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft_fftw3.cFFT(&S[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t2=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.cFFT(&S1[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i <= N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose > 1) {
            printf("%e <> %e <> %e\n", S2[i], S1[i], S[i]);
        }
    }
}

void test_cos_fftw3_double(void** s) {
    test_cos_fftw3<double>(s);
}

void test_cos_fftw3_float(void** s) {
    test_cos_fftw3<float>(s);
}
#endif

template<typename T>
void test_periodic_new2(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFT<T> ft_old(table, N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    vector<T> S1(N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft_old.pFFT_1(&S[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t2=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.pFFT_1(&S1[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose > 1) {
            printf("%e <> %e\n", S1[i], S[i]);
        }
    }
}

void test_periodic_new2_double(void** s) {
    test_periodic_new2<double>(s);
}

void test_periodic_new2_float(void** s) {
    test_periodic_new2<float>(s);
}

#ifdef HAVE_FFTW3
template<typename T>
void test_periodic_fftw3_pFFT_1(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFT_fftw3<T> ft_fftw3(N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    vector<T> S1(N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft_fftw3.pFFT_1(&S[0], &s1[0], 2.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t2=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.pFFT_1(&S1[0], &s1[0], 2.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose > 1) {
            printf("%e <> %e\n", S1[i], S[i]);
        }
    }
}

void test_periodic_fftw3_pFFT_1_double(void** s) {
    test_periodic_fftw3_pFFT_1<double>(s);
}

void test_periodic_fftw3_pFFT_1_float(void** s) {
    test_periodic_fftw3_pFFT_1<float>(s);
}

template<typename T>
void test_periodic_fftw3_pFFT(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFT_fftw3<T> ft_fftw3(N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    vector<T> S1(N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft_fftw3.pFFT(&S[0], &s1[0], 2.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t2=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.pFFT(&S1[0], &s1[0], 2.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose > 1) {
            printf("%e <> %e\n", S1[i], S[i]);
        }
    }
}

void test_periodic_fftw3_pFFT_double(void** s) {
    test_periodic_fftw3_pFFT<double>(s);
}

void test_periodic_fftw3_pFFT_float(void** s) {
    test_periodic_fftw3_pFFT<float>(s);
}
#endif

#ifdef HAVE_ONEMATH
template<typename T>
void test_periodic_sycl_pFFT_1(void** data) {
    sycl::queue q{ sycl::default_selector_v };
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFTSycl<T> ft_sycl(q, N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    using alloc_type = fdm::sycl_allocator<T>;
    alloc_type alloc(q);
    vector<T,alloc_type> S(N, alloc);
    vector<T,alloc_type> s(N, alloc);
    vector<T,alloc_type> s1(N, alloc);
    vector<T,alloc_type> S1(N, alloc);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft_sycl.pFFT_1(&S[0], &s1[0], 2.0).wait();
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t2=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.pFFT_1(&S1[0], &s1[0], 2.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose > 1) {
            printf("%e <> %e\n", S1[i], S[i]);
        }
    }
}

void test_periodic_sycl_pFFT_1_double(void** s) {
    test_periodic_sycl_pFFT_1<double>(s);
}

void test_periodic_sycl_pFFT_1_float(void** s) {
    test_periodic_sycl_pFFT_1<float>(s);
}

template<typename T>
void test_periodic_sycl_pFFT(void** data) {
    sycl::queue q{ sycl::default_selector_v };
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFTSycl<T> ft_sycl(q, N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    using alloc_type = fdm::sycl_allocator<T>;
    alloc_type alloc(q);
    vector<T, alloc_type> S(N, alloc);
    vector<T, alloc_type> s(N, alloc);
    vector<T, alloc_type> s1(N, alloc);
    vector<T, alloc_type> S1(N, alloc);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft_sycl.pFFT(&S[0], &s1[0], 2.0).wait();
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t2=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.pFFT(&S1[0], &s1[0], 2.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose > 1) {
            printf("%e <> %e\n", S1[i], S[i]);
        }
    }
}

void test_periodic_sycl_pFFT_double(void** s) {
    test_periodic_sycl_pFFT<double>(s);
}

void test_periodic_sycl_pFFT_float(void** s) {
    test_periodic_sycl_pFFT<float>(s);
}
#endif


template<typename T>
void test_complex(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    vector<T> S1(N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.pFFT_1(&S1[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.cpFFT(&S[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    if (verbose == 2) {
        for (int i = 0; i < N; i++) {
            printf("%f %f\n", S1[i], S[i]);
        }
    }
}

void test_complex_double(void** s) {
    test_complex<double>(s);
}

void test_complex_float(void** s) {
    test_complex<float>(s);
}

template<typename T>
void test_sin_omp(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFT_debug_omp<T> ft_omp(table, N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    vector<T> S1(N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.sFFT(&S1[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        s1 = s;
        auto t1 = steady_clock::now();
        ft_omp.sFFT(&S[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose == 2) {
            printf("%f %f\n", S1[i], S[i]);
        }
    }
}

void test_sin_omp_double(void** s) {
    test_sin_omp<double>(s);
}

void test_sin_omp_float(void** s) {
    test_sin_omp<float>(s);
}

template<typename T>
void test_cos_omp(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFT_debug_omp<T> ft_omp(table, N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N+1);
    vector<T> s(N+1);
    vector<T> s1(N+1);
    vector<T> S1(N+1);

    for (int i = 0; i < N+1; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.cFFT(&S1[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        s1 = s;
        auto t1 = steady_clock::now();
        ft_omp.cFFT(&S[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N+1; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose == 2) {
            printf("%f %f\n", S1[i], S[i]);
        }
    }
}

void test_cos_omp_double(void** s) {
    test_cos_omp<double>(s);
}

void test_cos_omp_float(void** s) {
    test_cos_omp<float>(s);
}

template<typename T>
void test_per_omp(void** data) {
    Config* c = static_cast<Config*>(*data);
    int N = c->get("test", "N", 256);
    int verbose = c->get("test", "verbose", 0);
    FFTTable<T> table(N);
    fdm::FFT<T> ft(table, N);
    fdm::FFT_debug_omp<T> ft_omp(table, N);
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-1, 1);
    vector<T> S(N);
    vector<T> s(N);
    vector<T> s1(N);
    vector<T> S1(N);

    for (int i = 0; i < N; i++) {
        s[i] = distribution(generator);
    }

    s1 = s;
    {
        auto t1 = steady_clock::now();
        ft.pFFT_1(&S1[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    s1 = s;
    {
        s1 = s;
        auto t1 = steady_clock::now();
        ft_omp.pFFT_1(&S[0], &s1[0], 1.0);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        if (verbose) {
            printf("t3=%f\n", interval.count());
        }
    }

    double tol = 1e-15;
    if constexpr(is_same<float,T>::value) {
        tol = 1e-3;
    }

    for (int i = 0; i < N; i++) {
        assert_float_equal(S1[i], S[i], tol);
        if (verbose == 2) {
            printf("%f %f\n", S1[i], S[i]);
        }
    }
}

void test_per_omp_double(void** s) {
    test_per_omp<double>(s);
}

void test_per_omp_float(void** s) {
    test_per_omp<float>(s);
}

int main(int argc, char** argv) {
    string config_fn = "ut_fft.ini";
    Config c;
    c.open(config_fn);
    c.rewrite(argc, argv);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_periodic),
        cmocka_unit_test(test_periodic_new_double),
        cmocka_unit_test(test_periodic_new_float),
        cmocka_unit_test(test_periodic_3d_double),
        cmocka_unit_test(test_periodic_3d_float),
        cmocka_unit_test(test_sin_double),
        cmocka_unit_test(test_sin_float),
        cmocka_unit_test(test_cos_double),
        cmocka_unit_test(test_cos_float),
        cmocka_unit_test(test_periodic_new_old_cmp),
        cmocka_unit_test_prestate(test_sin_new_double, &c),
        cmocka_unit_test_prestate(test_sin_new_float, &c),
        cmocka_unit_test_prestate(test_cos_new_double, &c),
        cmocka_unit_test_prestate(test_cos_new_float, &c),
        cmocka_unit_test_prestate(test_periodic_new2_double, &c),
        cmocka_unit_test_prestate(test_periodic_new2_float, &c),
        cmocka_unit_test_prestate(test_complex_float, &c),
        cmocka_unit_test_prestate(test_complex_double, &c),
#ifdef _OPENMP
        cmocka_unit_test_prestate(test_sin_omp_float, &c),
        cmocka_unit_test_prestate(test_sin_omp_double, &c),
        cmocka_unit_test_prestate(test_cos_omp_float, &c),
        cmocka_unit_test_prestate(test_cos_omp_double, &c),
        cmocka_unit_test_prestate(test_per_omp_float, &c),
        cmocka_unit_test_prestate(test_per_omp_double, &c),
#endif
#ifdef HAVE_FFTW3
        cmocka_unit_test_prestate(test_periodic_fftw3_pFFT_1_float, &c),
        cmocka_unit_test_prestate(test_periodic_fftw3_pFFT_1_double, &c),
        cmocka_unit_test_prestate(test_periodic_fftw3_pFFT_float, &c),
        cmocka_unit_test_prestate(test_periodic_fftw3_pFFT_double, &c),
        cmocka_unit_test_prestate(test_sin_fftw3_float, &c),
        cmocka_unit_test_prestate(test_sin_fftw3_double, &c),
        cmocka_unit_test_prestate(test_cos_fftw3_float, &c),
        cmocka_unit_test_prestate(test_cos_fftw3_double, &c),
#endif
#ifdef HAVE_ONEMATH
        cmocka_unit_test_prestate(test_periodic_sycl_pFFT_1_float, &c),
        cmocka_unit_test_prestate(test_periodic_sycl_pFFT_1_double, &c),
        cmocka_unit_test_prestate(test_periodic_sycl_pFFT_float, &c),
        cmocka_unit_test_prestate(test_periodic_sycl_pFFT_double, &c),
        cmocka_unit_test(test_periodic_3d_sycl_double),
        cmocka_unit_test(test_periodic_3d_sycl_float),
#endif
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
};
