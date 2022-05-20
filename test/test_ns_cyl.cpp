#include <string>
#include <vector>
#include <climits>
#include <cmath>

#include "tensor.h"
#include "matrix_plot.h"
#include "config.h"
#include "sparse.h"
#include "asp_misc.h"

using namespace std;
using namespace fdm;

template<typename T, bool check>
class NSCyl {
public:
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;
    using tensor = fdm::tensor<T,3,check,tensor_flags>;
    using matrix = fdm::tensor<T,2,check>;

    const double R, r;
    const double h1, h2;
    const double U0; // скорость вращения внутреннего цилиндра

    const double Re;
    const double dt;

    const int nr, nz, nphi;
    const double dr, dz, dphi;
    const double dr2, dz2, dphi2;

    tensor u /*r*/,v/*z*/,w/*phi*/;
    tensor p,x;
    tensor F,G,H,RHS;
    matrix psi; // срез по плоскости Oyz

    csr_matrix<T> P;
    bool P_initialized = false;

    umfpack_solver<T> solver;
    umfpack_solver<T> solver_stream; // для функции тока по срезу

    int time_index = 0;

    NSCyl(const Config& c)
        : R(c.get("ns", "R", M_PI))
        , r(c.get("ns", "r", M_PI/2))
        , h1(c.get("ns", "h1", 0))
        , h2(c.get("ns", "h2", 10))
        , U0(c.get("ns", "u0", 1))
        , Re(c.get("ns", "Re", 1.0))
        , dt(c.get("ns", "dt", 0.001))

        , nr(c.get("ns", "nr", 32))
        , nz(c.get("ns", "nz", 32))
        , nphi(c.get("ns", "nphi", 32))
        , dr((R-r)/nr), dz((h2-h1)/nz), dphi(2*M_PI/nphi)
        , dr2(dr*dr), dz2(dz*dz), dphi2(dphi*dphi)

          // phi, z, r
        , u{{0, nphi, 0, nz+1, 0, nr+1}} // check bounds
        , v{{0, nphi, 0, nz+1, 0, nr+1}} // check bounds
        , w{{0, nphi, 0, nz+1, 0, nr+1}} // check bounds
        , p({0, nphi, 0, nz+1, 0, nr+1})
        , x({0, nphi, 0, nz+1, 0, nr+1})
        , F({0, nphi-1, 1, nz, 1, nr}) // check bounds
        , G({0, nphi-1, 1, nz, 1, nr}) // check bounds
        , H({0, nphi-1, 1, nz, 1, nr}) // check bounds
        , RHS({0, nphi-1, 1, nz, 1, nr})
        , psi({1, nz, 1, nr})
    { }

    void step() {
        time_index++;
    }
    void plot() { }

private:
    void FGH() {
        // F
        for (int k = 1; k <= nz; k++) { // 3/2 ..
            for (int j = 0; j <= nr; j++) { // 1/2 ..
                // 17.9
                F[0][k][j] = u[0][k][j] + dt*(
                    (u[0][k][j+1]-2*u[0][k][j]+u[0][k][j-1])/Re/dr2+
                    (u[0][k-1][j]-2*u[0][k][j]+u[0][k+1][j])/Re/dz2-
                    (sq(0.5*(u[0][k][j]+u[0][k][j+1]))-sq(0.5*(u[k][j-1]+u[0][k][j])))/dr-
                    0.25*((u[0][k]  [j]+u[0][k+1][j])*(v[0][k]  [j+1]+v[0][k]  [j])-
                          (u[0][k-1][j]+u[0][k]  [j])*(v[0][k-1][j+1]+v[0][k-1][j])
                        )/dz);
            }
        }
        // G
        // H
    }

    void init_P() {
        if (P_initialized) {
            return;
        }

        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    int id = RHS.index({i,k,j});

                    P.add(id, RHS.index(i-1,k,j), 1/dphi2);

                    if (k > 1) {
                        P.add(id, RHS.index(i,k-1,j), 1/dz2);
                    }

                    if (j > 1) {
                        P.add(id, RHS.index(i,k,j-1), 1/dr2);
                    }

                    P.add(id, RHS.index(i,k,j), -2/dr2-2/dz2-2/dphi2);

                    if (j < nr) {
                        P.add(id, RHS.index(i,k,j+1, 1/dr2));
                    }

                    if (k < nz) {
                        P.add(id, RHS.index(i,k+1,j));
                    }

                    P.add(id, RHS.index(i+1,k,j), 1/dphi2);
                }
            }
        }
        P.close();

        solver = std::move(P);

        P_initialized = true;
    }
};

template<typename T, bool check>
void calc(const Config& c) {
    NSCyl<T, true> ns(c);

    const int steps = c.get("ns", "steps", 1);
    const int plot_interval = c.get("plot", "interval", 100);
    int i;

    ns.plot();
    for (i = 0; i < steps; i++) {
        ns.step();

        if ((i+1) % plot_interval == 0) {
            ns.plot();
        }
    }
}

// Флетчер, том 2, страница 398
int main(int argc, char** argv) {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    bool check = c.get("other", "check", 0) == 1;
    if (check) {
        calc<double,true>(c);
    } else {
        calc<double,false>(c);
    }

    return 0;
}
