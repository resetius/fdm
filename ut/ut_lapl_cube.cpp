#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>

#include "lapl_cube.h"
#include "config.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;
using namespace std;
using namespace std::chrono;
using namespace asp;

double ans(int i, int k, int j, double dz, double dy, double dx, double x1, double y1, double z1, double x2, double y2, double z2) {
    double x = x1+dx*j-dx/2;
    double y = y1+dy*k-dy/2;
    double z = z1+dz*i-dz/2;

    return sq(sin(x))+sq(cos(y))+sq(sin(z));
}

double rp(int i, int k, int j, double dz, double dy, double dx, double x1, double y1, double z1, double x2, double y2, double z2) {
    double x = x1+dx*j-dx/2;
    double y = y1+dy*k-dy/2;
    double z = z1+dz*i-dz/2;

    return 2*sq(sin(y))-2*sq(cos(y))
        -2*sq(sin(x))+2*sq(cos(x))
        -2*sq(sin(z))+2*sq(cos(z));
}

template<typename T>
void test_lapl_cube(void** data) {
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using tensor = fdm::tensor<T,3,check>;

    double x1 = c->get("test", "x1", 0.0);
    double y1 = c->get("test", "y1", 0.0);
    double z1 = c->get("test", "z1", 0.0);
    double x2 = c->get("test", "x2", 1.0);
    double y2 = c->get("test", "y2", 1.0);
    double z2 = c->get("test", "z2", 1.0);
    int nx = c->get("test", "nx", 31);
    int ny = c->get("test", "ny", 31);
    int nz = c->get("test", "nz", 31);
    int verbose = c->get("test", "verbose", 0);

    double dx = (x2-x1)/nx, dy = (y2-y1)/ny, dz = (z2-z1)/nz;

    LaplCube<T,check> lapl(
        dx, dy, dz,
        x2-x1+dx, y2-y1+dy, z2-z1+dz,
        nx, ny, nz);

    array<int,6> indices = {1,nz,1,ny,1,nx};
    tensor RHS(indices);
    tensor ANS(indices);

    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                RHS[i][k][j] = rp(i, k, j, dz, dy, dx, x1, y1, z1, x2, y2, z2);

                if (i <= 1) {
                    RHS[i][k][j] -= ans(i-1, k, j, dz, dy, dx,x1,y1,z1,x2,y2,z2)/dz/dz;
                }
                if (k <= 1) {
                    RHS[i][k][j] -= ans(i, k-1, j, dz, dy, dx,x1,y1,z1,x2,y2,z2)/dy/dy;
                }
                if (j <= 1) {
                    RHS[i][k][j] -= ans(i, k, j-1, dz, dy, dx,x1,y1,z1,x2,y2,z2)/dx/dx;
                }
                if (j >= nx) {
                    RHS[i][k][j] -= ans(i, k, j+1, dz, dy, dx,x1,y1,z1,x2,y2,z2)/dx/dx;
                }
                if (k >= ny) {
                    RHS[i][k][j] -= ans(i, k+1, j, dz, dy, dx,x1,y1,z1,x2,y2,z2)/dy/dy;
                }
                if (i >= nz) {
                    RHS[i][k][j] -= ans(i+1, k, j, dz, dy, dx,x1,y1,z1,x2,y2,z2)/dz/dz;
                }
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[1][1][1], &RHS[1][1][1]);
    auto t2 = steady_clock::now();

    double nrm = 0;
    double nrm1= 0;
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                double f = ans(i, k, j, dz, dy, dx, x1, y1, z1, x2, y2, z2);
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

    assert_true(nrm < 1e-4);
}

void test_lapl_cube_double(void** data) {
    test_lapl_cube<double>(data);
}

void test_lapl_cube_float(void** data) {
    test_lapl_cube<float>(data);
}

int main(int argc, char** argv) {
    string config_fn = "ut_lapl_cube.ini";
    Config c;
    c.open(config_fn);
    c.rewrite(argc, argv);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_lapl_cube_double, &c),
        cmocka_unit_test_prestate(test_lapl_cube_float, &c),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
