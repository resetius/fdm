#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>

#include "umfpack_solver.h"
#include "superlu_solver.h"
#include "lapl_cyl.h"
#include "config.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;
using namespace std;
using namespace std::chrono;
using namespace asp;

//(z-h0)*(z-h1)*(r-r0)*(r-R)*(sin(φ) + cos(φ))

double ans(int i, int k, int j, double dr, double dz, double dphi, double r0, double R, double h0, double h1) {
    double r = r0+dr*j-dr/2;
    double z = h0+dz*k-dz/2;
    double phi = dphi*(i+1)-dphi/2;
    double f = (z-h0)*(z-h1)*(r-r0)*(r-R)*(sin(phi) + cos(phi));
    return f;
}

//((r-r0)*(z-h0)*(z-h1)*(sin(φ)+cos(φ))+(r-R)*(z-h0)*(z-h1)*(sin(φ)+cos(φ))+2*r*(z-h0)*(z-h1)*(sin(φ)+cos(φ)))/r+2*(r-R)*(r-r0)*(sin(φ)+cos(φ))+((r-R)*(r-r0)*(z-h0)*(z-h1)*(-sin(φ)-cos(φ)))/r^2

double rp(int i, int k, int j, double dr, double dz, double dphi, double r0, double R, double h0, double h1) {
    double r = r0+dr*j-dr/2;
    double z = h0+dz*k-dz/2;
    double phi = dphi*(i+1)-dphi/2;
    double f = ((r-r0)*(z-h0)*(z-h1)*(sin(phi)+cos(phi))
                +(r-R)*(z-h0)*(z-h1)*(sin(phi)+cos(phi))
                +2*r*(z-h0)*(z-h1)*(sin(phi)+cos(phi)))/r
        +2*(r-R)*(r-r0)*(sin(phi)+cos(phi))
        +((r-R)*(r-r0)*(z-h0)*(z-h1)*(-sin(phi)-cos(phi)))/r/r;
    return f;
}

//(r-r0)*(r-R)*(sin(φ) + cos(φ)) + sin(z)^2+cos(z)^2
double ansp(int i, int k, int j, double dr, double dz, double dphi, double r0, double R, double h0, double h1) {
    double r = r0+dr*j-dr/2;
    double z = h0+dz*k-dz/2;
    double phi = dphi*(i+1)-dphi/2;
    double f = (r-r0)*(r-R)*(sin(phi) + cos(phi)) + sq(sin(z)) + sq(cos(z));
    return f;
}

double rpp(int i, int k, int j, double dr, double dz, double dphi, double r0, double R, double h0, double h1)
{
    double r = r0+dr*j-dr/2;
    //double z = h0+dz*k-dz/2;
    double phi = dphi*(i+1)-dphi/2;

    return ((-sin(phi)-cos(phi))*(r-R)*(r-r0))/r/r+
        ((sin(phi)+cos(phi))*(r-r0)+
         (sin(phi)+cos(phi))*(r-R)+
         (2*sin(phi)+2*cos(phi))*r)/r;
}

template<typename T,template<typename> class Solver>
void test_lapl_cyl_simple(void** data) {
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

    Config* c = static_cast<Config*>(*data);
    int nr = c->get("test", "nr", 16);
    int nz = c->get("test", "nz", 16);
    int nphi = c->get("test", "nphi", 16);
    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    LaplCyl3Simple<T, Solver, true> lapl(
        R, r0, h0, h1,
        nr, nz, nphi
        );
    double dr, dz, dphi;
    dr = lapl.dr; dz = lapl.dz; dphi = lapl.dphi;

    std::vector<int> indices = {0, nphi-1, 1, nz, 1, nr};
    tensor<T,3,check,tensor_flags> RHS(indices);
    tensor<T,3,check,tensor_flags> ANS(indices);

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;

                RHS[i][k][j] = rp(i, k, j, dr, dz, dphi, r0, R, h0, h1);

                if (k <= 1) {
                    RHS[i][k][j] -= ans(i,k-1,j,dr,dz,dphi, r0, R, h0, h1)/dz/dz;
                }
                if (j <= 1) {
                    RHS[i][k][j] -= (r-dr/2)/r*ans(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r*ans(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (k >= nz) {
                    RHS[i][k][j] -= ans(i,k+1,j,dr,dz,dphi,r0,R,h0,h1)/dz/dz;
                }
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[0][1][1], &RHS[0][1][1]);
    auto t2 = steady_clock::now();

    double nrm = 0.0;
    double nrm1 = 0.0;
    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double f = ans(i,k,j, dr, dz, dphi,r0,R,h0,h1);
                if (verbose > 1) {
                    printf("%e %e %e %e\n",
                           ANS[i][k][j], f,
                           ANS[i][k][j]/f,
                           std::abs(ANS[i][k][j]-f));
                }

                nrm = std::max(nrm, std::abs(ANS[i][k][j]-f));
                nrm1 = std::max(nrm1, std::abs(f));
            }
        }
    }
    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

    assert_true(nrm < 1e-3);
}

void test_lapl_cyl_simple_double(void** data) {
    test_lapl_cyl_simple<double,umfpack_solver>(data);
}

void test_lapl_cyl_simple_float(void** data) {
    test_lapl_cyl_simple<float,superlu_solver>(data);
}

template<typename T,template<typename> class Solver>
void test_lapl_cyl(void** data) {
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

    int nr = c->get("test", "nr", 32);
    int nz = c->get("test", "nz", 31);
    int nphi = c->get("test", "nphi", 32);
    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    double dr = (R-r0)/nr;
    double dz = (h1-h0)/nz;
    double dphi = 2*M_PI/nphi;

    LaplCyl3FFT1<T, Solver, true> lapl(dr, dz, r0-dr/2, R-r0+dr, h1-h0+dz, nr, nz, nphi);

    std::vector<int> indices = {0, nphi-1, 1, nz, 1, nr};
    tensor<T,3,check,tensor_flags> RHS(indices);
    tensor<T,3,check,tensor_flags> ANS(indices);

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;

                RHS[i][k][j] = rp(i, k, j, dr, dz, dphi, r0, R, h0, h1);

                if (k <= 1) {
                    RHS[i][k][j] -= ans(i,k-1,j,dr,dz,dphi, r0, R, h0, h1)/dz/dz;
                }
                if (j <= 1) {
                    RHS[i][k][j] -= (r-dr/2)/r*ans(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r*ans(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (k >= nz) {
                    RHS[i][k][j] -= ans(i,k+1,j,dr,dz,dphi,r0,R,h0,h1)/dz/dz;
                }
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[0][1][1], &RHS[0][1][1]);
    auto t2 = steady_clock::now();

    double nrm = 0.0;
    double nrm1 = 0.0;
    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double f = ans(i,k,j, dr, dz, dphi,r0,R,h0,h1);
                if (verbose > 1) {
                    printf("%e %e %e %e\n",
                           ANS[i][k][j], f,
                           ANS[i][k][j]/f,
                           std::abs(ANS[i][k][j]-f));
                }
                nrm = std::max(nrm, std::abs(ANS[i][k][j]-f));
                nrm1 = std::max(nrm1, std::abs(f));
            }
        }
    }
    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

    assert_true(nrm < 1e-3);
}

void test_lapl_cyl_double(void** data) {
    test_lapl_cyl<double,umfpack_solver>(data);
}

void test_lapl_cyl_float(void** data) {
    test_lapl_cyl<float,superlu_solver>(data);
}

template<typename T>
void test_lapl_cyl_zp(void** data) {
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic,tensor_flag::periodic>;

    int nr = c->get("test", "nr", 32);
    int nz = c->get("test", "nz", 32);
    int nphi = c->get("test", "nphi", 32);
    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    double dr = (R-r0)/nr;
    double dz = (h1-h0)/nz;
    double dphi = 2*M_PI/nphi;

    LaplCyl3FFT2<T,true,tensor_flag::periodic>
        lapl(dr, dz, r0-dr/2, R-r0+dr, h1-h0, nr, nz, nphi);

    std::vector<int> indices = {0, nphi-1, 0, nz-1, 1, nr};
    tensor<T,3,check,tensor_flags> RHS(indices);
    tensor<T,3,check,tensor_flags> ANS(indices);

    for (int i = 0; i < nphi; i++) {
        for (int k = 0; k < nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;

                RHS[i][k][j] = rpp(i, k, j, dr, dz, dphi, r0, R, h0, h1);

                if (j <= 1) {
                    RHS[i][k][j] -= (r-dr/2)/r*ansp(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r*ansp(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
            }
        }
    }

    auto t1 = steady_clock::now();
    //             phi z  r
    lapl.solve(&ANS[0][0][1], &RHS[0][0][1]);
    auto t2 = steady_clock::now();

    double nrm = 0.0;
    double nrm1 = 0.0;
    for (int i = 0; i < nphi; i++) {
        for (int k = 0; k < nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double f = ansp(i,k,j, dr, dz, dphi,r0,R,h0,h1);
                if (verbose > 1) {
                    printf("%e %e %e %e\n",
                           ANS[i][k][j], f,
                           ANS[i][k][j]/f,
                           std::abs(ANS[i][k][j]-f));
                }
                nrm = std::max(nrm, std::abs(ANS[i][k][j]-f));
                nrm1 = std::max(nrm1, std::abs(f));
            }
        }
    }
    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

    assert_true(nrm < 1e-3);
}

void test_lapl_cyl_zp_double(void** data) {
    test_lapl_cyl_zp<double>(data);
}

void test_lapl_cyl_zp_float(void** data) {
    test_lapl_cyl_zp<float>(data);
}

template<typename T>
T solve_lapl(Config* c, int nr, int nz, int nphi) {
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    double dr = (R-r0)/nr;
    double dz = (h1-h0)/nz;
    double dphi = 2*M_PI/nphi;

    LaplCyl3FFT2<T, true> lapl(dr, dz, r0-dr/2, R-r0+dr, h1-h0+dz, nr, nz, nphi);

    std::vector<int> indices = {0, nphi-1, 1, nz, 1, nr};
    tensor<T,3,check,tensor_flags> RHS(indices);
    tensor<T,3,check,tensor_flags> ANS(indices);

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;

                RHS[i][k][j] = rp(i, k, j, dr, dz, dphi, r0, R, h0, h1);

                if (k <= 1) {
                    RHS[i][k][j] -= ans(i,k-1,j,dr,dz,dphi, r0, R, h0, h1)/dz/dz;
                }
                if (j <= 1) {
                    RHS[i][k][j] -= (r-dr/2)/r*ans(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r*ans(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (k >= nz) {
                    RHS[i][k][j] -= ans(i,k+1,j,dr,dz,dphi,r0,R,h0,h1)/dz/dz;
                }
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[0][1][1], &RHS[0][1][1]);
    auto t2 = steady_clock::now();

    double nrm = 0.0;
    double nrm1 = 0.0;
    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double f = ans(i,k,j, dr, dz, dphi,r0,R,h0,h1);
                if (verbose > 1) {
                    printf("%e %e %e %e\n",
                           ANS[i][k][j], f,
                           ANS[i][k][j]/f,
                           std::abs(ANS[i][k][j]-f));
                }
                nrm = std::max(nrm, std::abs(ANS[i][k][j]-f));
                nrm1 = std::max(nrm1, std::abs(f));
            }
        }
    }
    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

    return nrm;
}

template<typename T>
void test_lapl_cyl_norm_decr(void** data) {
    Config* c = static_cast<Config*>(*data);

    int nr = c->get("test", "nr", 16);
    int nz = c->get("test", "nz", 15);
    int nphi = c->get("test", "nphi", 16);

    double nrm1 = solve_lapl<T>(c, nr, nz, nphi);
    double nrm2 = solve_lapl<T>(c, nr*2, (nz+1)*2-1, nphi*2);
    int verbose = c->get("test", "verbose", 0);
    if (verbose) {
        printf("nrm1/nrm2 = %e, %e %e\n", nrm1/nrm2, nrm1, nrm2);
    }
    assert_true(nrm1/nrm2 > 3.7);
}

void test_lapl_cyl_norm_decr_double(void** data) {
    test_lapl_cyl_norm_decr<double>(data);
}

void test_lapl_cyl_norm_decr_float(void** data) {
    test_lapl_cyl_norm_decr<float>(data);
}

void test_lapl_cyl_fft1_fft2_cmp(void** data) {
    using T = double;
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

    int nr = c->get("test", "nr", 32);
    int nz = c->get("test", "nz", 31);
    int nphi = c->get("test", "nphi", 32);
    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    double dr = (R-r0)/nr;
    double dz = (h1-h0)/nz;
    double dphi = 2*M_PI/nphi;

    LaplCyl3FFT1<T, umfpack_solver, true> lapl1(dr, dz, r0-dr/2, R-r0+dr, h1-h0+dz, nr, nz, nphi);
    LaplCyl3FFT2<T, true> lapl2(dr, dz, r0-dr/2, R-r0+dr, h1-h0+dz, nr, nz, nphi);

    std::vector<int> indices = {0, nphi-1, 1, nz, 1, nr};
    tensor<T,3,check,tensor_flags> RHS(indices);
    tensor<T,3,check,tensor_flags> ANS(indices);
    tensor<T,3,check,tensor_flags> ANS2(indices);

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;

                RHS[i][k][j] = rp(i, k, j, dr, dz, dphi, r0, R, h0, h1);

                if (k <= 1) {
                    RHS[i][k][j] -= ans(i,k-1,j,dr,dz,dphi, r0, R, h0, h1)/dz/dz;
                }
                if (j <= 1) {
                    RHS[i][k][j] -= (r-dr/2)/r*ans(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r*ans(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (k >= nz) {
                    RHS[i][k][j] -= ans(i,k+1,j,dr,dz,dphi,r0,R,h0,h1)/dz/dz;
                }
            }
        }
    }

    {
        auto t1 = steady_clock::now();
        lapl1.solve(&ANS[0][1][1], &RHS[0][1][1]);
        auto t2 = steady_clock::now();

        auto interval1 = duration_cast<duration<double>>(t2 - t1).count();
        if (verbose) {
            printf("%e\n", interval1);
        }
    }

    {
        auto t1 = steady_clock::now();
        lapl2.solve(&ANS2[0][1][1], &RHS[0][1][1]);
        auto t2 = steady_clock::now();

        auto interval1 = duration_cast<duration<double>>(t2 - t1).count();
        if (verbose) {
            printf("%e\n", interval1);
        }
    }

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                assert_float_equal(ANS[i][k][j], ANS2[i][k][j], 1e-15);
            }
        }
    }
}

int main(int argc, char** argv) {
    string config_fn = "ut_lapl.ini";
    Config c;
    c.open(config_fn);
    c.rewrite(argc, argv);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_lapl_cyl_simple_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_simple_float, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_norm_decr_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_norm_decr_float, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_float, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_zp_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_zp_float, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_fft1_fft2_cmp, &c),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
