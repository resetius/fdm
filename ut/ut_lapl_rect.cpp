#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>

#include "lapl_rect.h"
#include "config.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;
using namespace std;
using namespace std::chrono;
using namespace asp;

double ans(int k, int j, double dy, double dx, double x1, double y1, double x2, double y2) {
    double x = x1+dx*j-dx/2;
    double y = y1+dy*k-dy/2;

    return sq(sin(x))+sq(cos(y));
}

double rp(int k, int j, double dy, double dx, double x1, double y1, double x2, double y2) {
    double x = x1+dx*j-dx/2;
    double y = y1+dy*k-dy/2;

    return 2*sq(sin(y))-2*sq(cos(y))-2*sq(sin(x))+2*sq(cos(x));
}

double ans0(int k, int j, double dy, double dx, double x1, double y1, double x2, double y2) {
    double x = x1+dx*j-dx/2;
    double y = y1+dy*k-dy/2;

    return (x-x1)*(x-x2)*(y-y1)*(y-y2);
}

double rp0(int k, int j, double dy, double dx, double x1, double y1, double x2, double y2) {
    double x = x1+dx*j-dx/2;
    double y = y1+dy*k-dy/2;
    return 2*(y-y1)*(y-y2)+2*(x-x1)*(x-x2);
}

template<typename T>
void test_lapl_rect(void** data) {
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using matrix = tensor<T,2,check>;

    double x1 = c->get("test", "x1", 0.0);
    double y1 = c->get("test", "y1", 0.0);
    double x2 = c->get("test", "x2", 1.0);
    double y2 = c->get("test", "y2", 1.0);
    int nx = c->get("test", "nx", 31);
    int ny = c->get("test", "ny", 31);
    int verbose = c->get("test", "verbose", 0);

    double dx = (x2-x1)/nx, dy = (y2-y1)/ny;

    LaplRect<T,check> lapl(dx, dy, x2-x1+dx, y2-y1+dy, nx, ny);

    vector<int> indices = {1,ny,1,nx};
    matrix RHS(indices);
    matrix ANS(indices);

    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            RHS[k][j] = rp(k, j, dy, dx, x1, y1, x2, y2);

            if (k <= 1) {
                RHS[k][j] -= ans(k-1, j, dy, dx, x1, y1, x2, y2)/dy/dy;
            }
            if (j <= 1) {
                RHS[k][j] -= ans(k, j-1, dy, dx, x1, y1, x2, y2)/dx/dx;
            }
            if (j >= nx) {
                RHS[k][j] -= ans(k, j+1, dy, dx, x1, y1, x2, y2)/dx/dx;
            }
            if (k >= ny) {
                RHS[k][j] -= ans(k+1, j, dy, dx, x1, y1, x2, y2)/dy/dy;
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[1][1], &RHS[1][1]);
    auto t2 = steady_clock::now();

    double nrm = 0;
    double nrm1= 0;
    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            double f = ans(k, j, dy, dx, x1, y1, x2, y2);
            if (verbose > 1) {
                printf("%e %e %e %e\n",
                       ANS[k][j], f,
                       ANS[k][j]/f,
                       std::abs(ANS[k][j]-f));
            }

            nrm = std::max(nrm, std::abs(ANS[k][j]-f));
            nrm1 = std::max(nrm1, std::abs(f));
        }
    }

    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

    assert_true(nrm < 3e-3);
}

void test_lapl_rect_double(void** data) {
    test_lapl_rect<double>(data);
}

void test_lapl_rect_float(void** data) {
    test_lapl_rect<float>(data);
}

template<typename T>
T solve_lapl(Config* c, int nx, int ny) {
    constexpr bool check = true;
    using matrix = tensor<T,2,check>;

    double x1 = c->get("test", "x1", 0.0);
    double y1 = c->get("test", "y1", 0.0);
    double x2 = c->get("test", "x2", 1.0);
    double y2 = c->get("test", "y2", 1.0);
    int verbose = c->get("test", "verbose", 0);

    double dx = (x2-x1)/nx, dy = (y2-y1)/ny;

    LaplRect<T,check> lapl(dx, dy, x2-x1+dx, y2-y1+dy, nx, ny);

    vector<int> indices = {1,ny,1,nx};
    matrix RHS(indices);
    matrix ANS(indices);

    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            RHS[k][j] = rp(k, j, dy, dx, x1, y1, x2, y2);

            if (k <= 1) {
                RHS[k][j] -= ans(k-1, j, dy, dx, x1, y1, x2, y2)/dy/dy;
            }
            if (j <= 1) {
                RHS[k][j] -= ans(k, j-1, dy, dx, x1, y1, x2, y2)/dx/dx;
            }
            if (j >= nx) {
                RHS[k][j] -= ans(k, j+1, dy, dx, x1, y1, x2, y2)/dx/dx;
            }
            if (k >= ny) {
                RHS[k][j] -= ans(k+1, j, dy, dx, x1, y1, x2, y2)/dy/dy;
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[1][1], &RHS[1][1]);
    auto t2 = steady_clock::now();

    double nrm = 0;
    double nrm1= 0;
    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            double f = ans(k, j, dy, dx, x1, y1, x2, y2);
            if (verbose > 1) {
                printf("%e %e %e %e\n",
                       ANS[k][j], f,
                       ANS[k][j]/f,
                       std::abs(ANS[k][j]-f));
            }

            nrm = std::max(nrm, std::abs(ANS[k][j]-f));
            nrm1 = std::max(nrm1, std::abs(f));
        }
    }

    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    return nrm;
}

template<typename T>
void test_lapl_rect_norm_decr(void** data) {
    Config* c = static_cast<Config*>(*data);

    int nx = c->get("test", "nx", 15);
    int ny = c->get("test", "ny", 15);

    double nrm1 = solve_lapl<T>(c, nx, ny);
    double nrm2 = solve_lapl<T>(c, (nx+1)*2-1, (ny+1)*2-1);
    int verbose = c->get("test", "verbose", 0);
    if (verbose) {
        printf("nrm1/nrm2 = %e, %e %e\n", nrm1/nrm2, nrm1, nrm2);
    }
    assert_true(nrm1/nrm2 > 3.7);
}

void test_lapl_rect_norm_decr_double(void** data) {
    test_lapl_rect_norm_decr<double>(data);
}

void test_lapl_rect_norm_decr_float(void** data) {
    test_lapl_rect_norm_decr<float>(data);
}

template<typename T>
void test_lapl_rect_ex(void** data) {
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using matrix = tensor<T,2,check>;

    double x1 = c->get("test", "x1", 0.0);
    double y1 = c->get("test", "y1", 0.0);
    double x2 = c->get("test", "x2", 1.0);
    double y2 = c->get("test", "y2", 1.0);
    int nx = c->get("test", "nx", 31);
    int ny = c->get("test", "ny", 31);
    int verbose = c->get("test", "verbose", 0);

    double dx = (x2-x1)/nx, dy = (y2-y1)/ny;

    LaplRect<T,check> lapl(dx, dy, x2-x1+dx, y2-y1+dy, nx, ny);

    vector<int> indices = {1,ny,1,nx};
    matrix RHS(indices);
    matrix ANS(indices);

    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            RHS[k][j] = rp0(k, j, dy, dx, x1, y1, x2, y2);

            if (k <= 1) {
                RHS[k][j] -= ans0(k-1, j, dy, dx, x1, y1, x2, y2)/dy/dy;
            }
            if (j <= 1) {
                RHS[k][j] -= ans0(k, j-1, dy, dx, x1, y1, x2, y2)/dx/dx;
            }
            if (j >= nx) {
                RHS[k][j] -= ans0(k, j+1, dy, dx, x1, y1, x2, y2)/dx/dx;
            }
            if (k >= ny) {
                RHS[k][j] -= ans0(k+1, j, dy, dx, x1, y1, x2, y2)/dy/dy;
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[1][1], &RHS[1][1]);
    auto t2 = steady_clock::now();

    double nrm = 0;
    double nrm1= 0;
    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            double f = ans0(k, j, dy, dx, x1, y1, x2, y2);
            if (verbose > 1) {
                printf("%e %e %e %e\n",
                       ANS[k][j], f,
                       ANS[k][j]/f,
                       std::abs(ANS[k][j]-f));
            }

            nrm = std::max(nrm, std::abs(ANS[k][j]-f));
            nrm1 = std::max(nrm1, std::abs(f));
        }
    }

    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

    if constexpr(is_same<T,double>::value) {
        assert_true(nrm < 1e-14);
    } else {
        assert_true(nrm < 1e-5);
    }
}

void test_lapl_rect_ex_double(void** data) {
    test_lapl_rect_ex<double>(data);
}

void test_lapl_rect_ex_float(void** data) {
    test_lapl_rect_ex<float>(data);
}

void test_lapl_rect_fft1_fft2_cmp(void** data) {
    using T = double;
    Config* c = static_cast<Config*>(*data);

    constexpr bool check = true;
    using matrix = tensor<T,2,check>;

    double x1 = c->get("test", "x1", 0.0);
    double y1 = c->get("test", "y1", 0.0);
    double x2 = c->get("test", "x2", 1.0);
    double y2 = c->get("test", "y2", 1.0);
    int nx = c->get("test", "nx", 511);
    int ny = c->get("test", "ny", 511);

    int verbose = c->get("test", "verbose", 0);

    double dx = (x2-x1)/nx, dy = (y2-y1)/ny;

    LaplRect<T,check> lapl(dx, dy, x2-x1+dx, y2-y1+dy, nx, ny);

    vector<int> indices = {1,ny,1,nx};
    matrix RHS(indices);
    matrix ANS(indices);
    matrix ANS2(indices);

    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            RHS[k][j] = rp(k, j, dy, dx, x1, y1, x2, y2);

            if (k <= 1) {
                RHS[k][j] -= ans(k-1, j, dy, dx, x1, y1, x2, y2)/dy/dy;
            }
            if (j <= 1) {
                RHS[k][j] -= ans(k, j-1, dy, dx, x1, y1, x2, y2)/dx/dx;
            }
            if (j >= nx) {
                RHS[k][j] -= ans(k, j+1, dy, dx, x1, y1, x2, y2)/dx/dx;
            }
            if (k >= ny) {
                RHS[k][j] -= ans(k+1, j, dy, dx, x1, y1, x2, y2)/dy/dy;
            }
        }
    }

    {
        auto t1 = steady_clock::now();
        lapl.solve1(&ANS[1][1], &RHS[1][1]);
        auto t2 = steady_clock::now();

        auto interval1 = duration_cast<duration<double>>(t2 - t1).count();
        if (verbose) {
            printf("%e\n", interval1);
        }
    }

    {
        auto t1 = steady_clock::now();
        lapl.solve2(&ANS2[1][1], &RHS[1][1]);
        auto t2 = steady_clock::now();

        auto interval1 = duration_cast<duration<double>>(t2 - t1).count();
        if (verbose) {
            printf("%e\n", interval1);
        }

        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                assert_float_equal(ANS[k][j], ANS2[k][j], 1e-15);
            }
        }
    }
}

int main(int argc, char** argv) {
    string config_fn = "ut_lapl_rect.ini";
    Config c;
    c.open(config_fn);
    c.rewrite(argc, argv);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_lapl_rect_double, &c),
        cmocka_unit_test_prestate(test_lapl_rect_float, &c),
        cmocka_unit_test_prestate(test_lapl_rect_norm_decr_double, &c),
        cmocka_unit_test_prestate(test_lapl_rect_norm_decr_float, &c),
        cmocka_unit_test_prestate(test_lapl_rect_ex_double, &c),
        cmocka_unit_test_prestate(test_lapl_rect_ex_float, &c),
        cmocka_unit_test_prestate(test_lapl_rect_fft1_fft2_cmp, &c),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
