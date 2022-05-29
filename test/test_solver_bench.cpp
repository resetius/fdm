#include <string>
#include <vector>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdio>

#include "tensor.h"
#include "matrix_plot.h"
#include "config.h"
#include "sparse.h"
#include "umfpack_solver.h"
#include "gmres_solver.h"
#include "asp_misc.h"

using namespace std;
using namespace fdm;

using asp::format;
using asp::sq;

template<typename T, typename Solver, bool check=true>
class Benchmark {
public:
    using tensor = fdm::tensor<T,3,check>;

    const double x1,y1,z1;
    const double x2,y2,z2;

    const int nx, ny, nz;
    const double dx, dy, dz;
    const double dx2, dy2, dz2;

    tensor x,RHS;

    csr_matrix<T> P;

    Solver solver;

    Benchmark(const Config& c)
        : x1(c.get("ns", "x1", -M_PI))
        , y1(c.get("ns", "y1", -M_PI))
        , z1(c.get("ns", "z1", -M_PI))
        , x2(c.get("ns", "x2",  M_PI))
        , y2(c.get("ns", "y2",  M_PI))
        , z2(c.get("ns", "z2",  M_PI))

        , nx(c.get("ns", "nx", 32))
        , ny(c.get("ns", "nx", 32))
        , nz(c.get("ns", "nz", 32))

        , dx((x2-x1)/nx), dy((y2-y1)/ny), dz((z2-z1)/nz)
        , dx2(dx*dx), dy2(dy*dy), dz2(dz*dz)

        , x({1, nz, 1, ny, 1, nx})
        , RHS({1, nz, 1, ny, 1, nx})
    {
        init_matrix();
        init_rhs();
    }

    void init_matrix() {
        for (int i = 1; i <= nz; i++) {
            for (int k = 1; k <= ny; k++) {
                for (int j = 1; j <= nx; j++) {
                    int id = RHS.index({i,k,j});
                    if (i > 1) {
                        P.add(id, RHS.index({i-1,k,j}), 1/dz2);
                    }

                    if (k > 1) {
                        P.add(id, RHS.index({i,k-1,j}), 1/dy2);
                    }

                    if (j > 1) {
                        P.add(id, RHS.index({i,k,j-1}), 1/dx2);
                    }

                    P.add(id, RHS.index({i,k,j}), -2/dx2-2/dy2-2/dz2);

                    if (j < nx) {
                        P.add(id, RHS.index({i,k,j+1}), 1/dx2);
                    }

                    if (k < ny) {
                        P.add(id, RHS.index({i,k+1,j}), 1/dy2);
                    }

                    if (i < nz) {
                        P.add(id, RHS.index({i+1,k,j}), 1/dz2);
                    }
                }
            }
        }
        P.close();

        solver = std::move(P);
    }

    void init_rhs() {
        unsigned int seed = 1;
        for (int i = 1; i <= nz; i++) {
            for (int k = 1; k <= ny; k++) {
                for (int j = 1; j <= nx; j++) {
                    RHS[i][k][j] = 0.1 * (double) rand_r(&seed) / (double)RAND_MAX;
                }
            }
        }
    }

    void run(int iterations) {
        for (int i = 0; i < iterations; i++) {
            solver.solve(&x[1][1][1], &RHS[1][1][1]);
        }
    }

};

template<typename T, typename Solver>
void calc(const Config& c, const char* name) {
    using namespace std::chrono;

    {
        Benchmark<T, Solver> bench(c);
        auto t1 = steady_clock::now();
        bench.run(100);
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        printf("%s: '%f' seconds\n", name, interval.count());
    }
}

int main(int argc, char** argv) {
    string config_fn = "benchamrk.ini";

    Config c;
    calc<double, umfpack_solver<double>>(c, "umfpack");
    calc<double, gmres_solver<double>>(c, "gmres");

    return 0;
}
