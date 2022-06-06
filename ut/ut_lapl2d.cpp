#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>

#include "lapl2d.h"
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

    return (x-x1)*(x-x2)*(y-y1)*(y-y2);
}

double rp(int k, int j, double dy, double dx, double x1, double y1, double x2, double y2) {
    double x = x1+dx*j-dx/2;
    double y = y1+dy*k-dy/2;
    return 2*(y-y1)*(y-y2)+2*(x-x1)*(x-x2);
}

template<typename T>
void test_lapl2d(void** data) {
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using matrix = tensor<T,2,check>;

    double x1 = c->get("test", "x1", 0.0);
    double y1 = c->get("test", "y1", 0.0);
    double x2 = c->get("test", "x2", 1.0);
    double y2 = c->get("test", "y2", 1.0);
    int nx = c->get("test", "nx", 4);
    int ny = c->get("test", "ny", 3);
    int verbose = c->get("test", "verbose", 0);

    Lapl2d<T,check> lapl(x1, y1, x2, y2, nx, ny);
    double dx, dy;
    dx = lapl.dx; dy = lapl.dy;

    vector<int> indices = {1,ny,1,nx};
    matrix RHS(indices);
    matrix ANS(indices);

    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            RHS[k][j] = rp(k, j, dy, dx, x1, y1, x2, y2);
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
}

void test_lapl2d_double(void** data) {
    test_lapl2d<double>(data);
}

void test_lapl2d_float(void** data) {
    test_lapl2d<float>(data);
}

int main(int argc, char** argv) {
    string config_fn = "ut_lapl2d.ini";
    Config c;
    c.open(config_fn);
    c.rewrite(argc, argv);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_lapl2d_double, &c),
        //cmocka_unit_test_prestate(test_lapl2d_float, &c)
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
