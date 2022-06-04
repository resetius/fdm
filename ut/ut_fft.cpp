#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <random>
#include <vector>
#include <type_traits>

#include "asp_fft.h"

extern "C" {
#include <cmocka.h>
}

using namespace std;

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

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_periodic)
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
};
