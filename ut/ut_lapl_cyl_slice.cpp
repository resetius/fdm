#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>

#include "lapl_rect.h"
#include "config.h"
#include "superlu_solver.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;
using namespace std;
using namespace std::chrono;
using namespace asp;

double ans_z(int i, int j, double dr, double dphi, double r0, double R)
{
    double r = r0+dr*j-dr/2;
    double phi = dphi*(i+1)-dphi/2;
    double z = 1; double h0 = 0; double h1 = 5;
    double f = (z-h0)*(z-h1)*(r-r0)*(r-R)*(sin(phi) + cos(phi));
    return f;
}

double rp_z(int i, int j, double dr, double dphi, double r0, double R)
{
    double r = r0+dr*j-dr/2;
    double phi = dphi*(i+1)-dphi/2;
    double z = 1; double h0 = 0; double h1 = 5;
    double f = ((r-r0)*(z-h0)*(z-h1)*(sin(phi)+cos(phi))
                +(r-R)*(z-h0)*(z-h1)*(sin(phi)+cos(phi))
                +2*r*(z-h0)*(z-h1)*(sin(phi)+cos(phi)))/r
        +2*(r-R)*(r-r0)*(sin(phi)+cos(phi))
        +((r-R)*(r-r0)*(z-h0)*(z-h1)*(-sin(phi)-cos(phi)))/r/r;
    return f;
}

template<typename T>
void test_lapl_cyl_slice_z(void** data) {
    // phi, r
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;
    using matrix = tensor<T,2,check,tensor_flags>;
    double r0 = c->get("test", "r0", M_PI/2);
    double R = c->get("test", "R", M_PI);
    int nr = c->get("test", "nr", 31);
    int nphi = c->get("test", "nphi", 32);
    int verbose = c->get("test", "verbose", 0);
    double dr = (R-r0)/nr;
    double dphi = 2*M_PI/nphi;
    double dphi2 = dphi*dphi; double dr2 = dr*dr;
    LaplRect<T,check,tensor_flags> lapl(dr, dphi, R-r0+dr, 2*M_PI, nr, nphi);
    for (int j = 1; j <= nr; j++) {
        double r = r0+j*dr-dr/2;
        lapl.lm_y_scale[j] = 1./r/r;
        lapl.U_scale[j] = (r+dr/2)/r;
        lapl.L_scale[j] = (r-dr/2)/r;
    }

    vector<int> indices = {0,nphi-1,1,nr};
    matrix RHS_z(indices);
    matrix ANS(indices);
    matrix ANS2(indices);
    matrix ANS3(indices);

    for (int i = 0; i < nphi; i++) {
        for (int j = 1; j <= nr; j++) {
            double r = r0+dr*j-dr/2;

            RHS_z[i][j] = rp_z(i, j, dr, dphi, r0, R);

            if (j <= 1) {
                RHS_z[i][j] -= (r-dr/2)/r*ans_z(i, j-1, dr, dphi, r0, R)/dr/dr;
            }
            if (j >= nr) {
                RHS_z[i][j] -= (r+dr/2)/r*ans_z(i, j+1, dr, dphi, r0, R)/dr/dr;
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[0][1], &RHS_z[0][1]);
    auto t2 = steady_clock::now();

    double nrm = 0;
    double nrm1= 0;

    for (int i = 0; i < nphi; i++) {
        for (int j = 1; j <= nr; j++) {
            double f = ans_z(i, j, dr, dphi, r0, R);
            if (verbose > 1) {
                printf("%e %e %e %e\n",
                       ANS[i][j], f,
                       ANS[i][j]/f,
                       std::abs(ANS[i][j]-f));
            }

            nrm = std::max(nrm, std::abs(ANS[i][j]-f));
            nrm1 = std::max(nrm1, std::abs(f));
        }
    }

    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

// TODO: smth wrong
//    assert_true(nrm < 3e-3);

    csr_matrix<T> P_z;
    superlu_solver<T> solver;
    for (int i = 0; i < nphi; i++) {
        for (int j = 1; j <= nr; j++) {
            int id = RHS_z.index({i,j});
            double r = r0+j*dr-dr/2;
            double r2 = r*r;
            double rm = 2;

            P_z.add(id, RHS_z.index({i-1,j}), 1/dphi2/r2);

            if (j > 1) {
                P_z.add(id, RHS_z.index({i,j-1}), (r-0.5*dr)/dr2/r);
            }

            P_z.add(id, RHS_z.index({i,j}), -rm/dr2-2/dphi2/r2);

            if (j < nr) {
                P_z.add(id, RHS_z.index({i,j+1}), (r+0.5*dr)/dr2/r);
            }

            P_z.add(id, RHS_z.index({i+1,j}), 1/dphi2/r2);
        }
    }
    P_z.close();
    P_z.sort_rows();

    solver = std::move(P_z);

    t1 = steady_clock::now();
    solver.solve(&ANS2[0][1], &RHS_z[0][1]);
    t2 = steady_clock::now();

    interval = duration_cast<duration<double>>(t2 - t1);
    if (verbose) {
        printf("It took me '%f' seconds\n", interval.count());
    }

    double tol = 1e-15;
    if constexpr(is_same<T,float>::value) {
        tol = 1e-4;
    }
    for (int i = 0; i < nphi; i++) {
        for (int j = 1; j <= nr; j++) {
            assert_float_equal(ANS2[i][j], ANS[i][j], tol);
        }
    }
}

void test_lapl_cyl_slice_z_double(void** data) {
    test_lapl_cyl_slice_z<double>(data);
}

void test_lapl_cyl_slice_z_float(void** data) {
    test_lapl_cyl_slice_z<float>(data);
}

int main(int argc, char** argv) {
    string config_fn = "ut_lapl_cyl_slice.ini";
    Config c;
    c.open(config_fn);
    c.rewrite(argc, argv);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_lapl_cyl_slice_z_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_slice_z_float, &c),
    };

    return cmocka_run_group_tests(tests, NULL, NULL);
}
