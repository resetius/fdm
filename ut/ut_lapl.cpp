#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>

#include "umfpack_solver.h"
#include "lapl.h"
#include "config.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;
using namespace std;
using namespace asp;

//(z-h0)*(z-h1)*(r-r0)*(r-R)*(sin(φ) + cos(φ))

double ans(int i, int k, int j, double dr, double dz, double dphi, double r0, double R, double h0, double h1) {
    double r = r0+dr*j-dr/2;
    double z = h0+dz*k-dz/2;
    double φ = dphi*(i+1)-dphi/2;
    double f = (z-h0)*(z-h1)*(r-r0)*(r-R)*(sin(φ) + cos(φ));
    return f;
}

//((r-r0)*(z-h0)*(z-h1)*(sin(φ)+cos(φ))+(r-R)*(z-h0)*(z-h1)*(sin(φ)+cos(φ))+2*r*(z-h0)*(z-h1)*(sin(φ)+cos(φ)))/r+2*(r-R)*(r-r0)*(sin(φ)+cos(φ))+((r-R)*(r-r0)*(z-h0)*(z-h1)*(-sin(φ)-cos(φ)))/r^2

double rp(int i, int k, int j, double dr, double dz, double dphi, double r0, double R, double h0, double h1) {
    double r = r0+dr*j-dr/2;
    double z = h0+dz*k-dz/2;
    double φ = dphi*(i+1)-dphi/2;
    double f = ((r-r0)*(z-h0)*(z-h1)*(sin(φ)+cos(φ))
                +(r-R)*(z-h0)*(z-h1)*(sin(φ)+cos(φ))
                +2*r*(z-h0)*(z-h1)*(sin(φ)+cos(φ)))/r
        +2*(r-R)*(r-r0)*(sin(φ)+cos(φ))
        +((r-R)*(r-r0)*(z-h0)*(z-h1)*(-sin(φ)-cos(φ)))/r/r;
    return f;
}

void test_lapl_cyl_simple(void** data) {
    constexpr bool check = true;
    using T = double;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

    Config* c = static_cast<Config*>(*data);
    int nr = c->get("test", "nr", 32);
    int nz = c->get("test", "nz", 32);
    int nphi = c->get("test", "nphi", 32);
    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    LaplCyl3Simple<double, umfpack_solver, true> lapl(
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
                    RHS[i][k][j] -= (r-dr/2)/2*ans(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/2*ans(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (k >= nz) {
                    RHS[i][k][j] -= ans(i,k+1,j,dr,dz,dphi,r0,R,h0,h1)/dz/dz;
                }
            }
        }
    }

    lapl.solve(&ANS[0][1][1], &RHS[0][1][1]);

    double nrm = 0.0;
    double nrm1 = 0.0;
    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double f = ans(i,k,j, dr, dz, dphi,r0,R,h0,h1);
                if (verbose) {
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
    if (verbose) {
        printf("%e\n", nrm);
    }

    assert_true(nrm < 5e-2);
}

void test_lapl_cyl(void** data) {
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using T = double;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

    int nr = c->get("test", "nr", 32);
    int nz = c->get("test", "nz", 32);
    int nphi = c->get("test", "nphi", 32);
    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    LaplCyl3<double, umfpack_solver, true> lapl(
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
                    RHS[i][k][j] -= (r-dr/2)/2*ans(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/2*ans(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (k >= nz) {
                    RHS[i][k][j] -= ans(i,k+1,j,dr,dz,dphi,r0,R,h0,h1)/dz/dz;
                }
            }
        }
    }

    lapl.solve(&ANS[0][1][1], &RHS[0][1][1]);

    double nrm = 0.0;
    double nrm1 = 0.0;
    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double f = ans(i,k,j, dr, dz, dphi,r0,R,h0,h1);
                if (verbose) {
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
    if (verbose) {
        printf("%e\n", nrm);
    }
}

int main(int argc, char** argv) {
    string config_fn = "ut_lapl.ini";
    Config c;
    c.open(config_fn);
    c.rewrite(argc, argv);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_lapl_cyl_simple, &c),
        cmocka_unit_test_prestate(test_lapl_cyl, &c)
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
