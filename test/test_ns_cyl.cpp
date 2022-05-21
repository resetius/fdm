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

using asp::format;
using asp::sq;

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
    matrix ui, vi; // срез по плоскости Oyz

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
        , u{{0, nphi-1, 0, nz+1, -1, nr+1}} // check bounds
        , v{{0, nphi-1, -1, nz+1, 0, nr+1}} // check bounds
        , w{{0, nphi-1, 0, nz+1, 0, nr+1}} // check bounds
        , p({0, nphi-1, 0, nz+1, 0, nr+1})
        , x({0, nphi-1, 0, nz+1, 0, nr+1})
        , F({0, nphi-1, 1, nz, 0, nr}) // check bounds
        , G({0, nphi-1, 0, nz, 1, nr}) // check bounds
        , H({0, nphi-1, 1, nz, 1, nr}) // check bounds
        , RHS({0, nphi-1, 1, nz, 1, nr})
        , psi({1, nz, 1, nr})
        , ui({1, nz, 1, nr}) // inner u
        , vi({1, nz, 1, nr}) // inner v
    { }

    void step() {
        init_bound();
        FGH();
        poisson();
        update_uvwp();
        time_index++;
    }

    void plot() {
        update_uvi();
        matrix_plotter plotter(matrix_plotter::settings()
                               .sub(2, 2)
                               .devname("pngcairo")
                               .fname(format("step_%07d.png", time_index)));
        plotter.plot(matrix_plotter::page()
                     .scalar(ui)
                     .levels(10)
                     .tlabel(format("U (t=%.1e, |max|=%.1e)", dt*time_index, ui.maxabs()))
                     .bounds(r+dr/2, h1+dz/2, r-dr/2, h2-dz/2));
        plotter.plot(matrix_plotter::page()
                     .scalar(vi)
                     .levels(10)
                     .tlabel(format("V (t=%.1e, |max|=%.1e)", dt*time_index, vi.maxabs()))
                     .bounds(r+dr/2, h1+dz/2, r-dr/2, h2-dz/2));
        plotter.plot(matrix_plotter::page()
                     .vector(ui, vi)
                     .levels(10)
                     .tlabel(format("UV (%.1e)", dt*time_index))
                     .bounds(r+dr/2, h1+dz/2, r-dr/2, h2-dz/2));
    }

private:
    void init_bound() {
    }

    void FGH() {
        // F
        for (int i = 1; i <= nphi; i++) {
            for (int k = 1; k <= nz; k++) { // 3/2 ..
                for (int j = 0; j <= nr; j++) { // 1/2 ..
                    // 17.9
                    F[i][k][j] = u[i][k][j] + dt*(
                        (u[i][k][j+1]-2*u[i][k][j]+u[i][k][j-1])/Re/dr2+
                        (u[i][k-1][j]-2*u[i][k][j]+u[i][k+1][j])/Re/dz2+
                        (u[i-1][k][j]-2*u[i][k][j]+u[i+1][k][j])/Re/dphi2-
                        (sq(0.5*(u[i][k][j]+u[i][k][j+1]))-sq(0.5*(u[i][k][j-1]+u[i][k][j])))/dr-

                        0.25*((u[i][k]  [j]+u[i][k+1][j])*(v[i][k]  [j+1]+v[i][k]  [j])-
                              (u[i][k-1][j]+u[i][k]  [j])*(v[i][k-1][j+1]+v[i][k-1][j])
                            )/dz-

                        0.25*((u[i]  [k][j]+u[i+1][k][j])*(w[i]  [k][j+1]+w[i]  [k][j])-
                              (u[i-1][k][j]+u[i]  [k][j])*(w[i-1][k][j+1]+w[i-1][k][j])
                            )/dphi);
                }
            }
        }
        // G
        for (int i = 1; i <= nphi; i++) {
            for (int k = 0; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    // 17.11
                    G[i][k][j] = v[i][k][j] + dt*(
                        (v[i][k][j+1]-2*v[i][k][j]+v[i][k][j-1])/Re/dr2+
                        (v[i][k+1][j]-2*v[i][k][j]+v[i][k-1][j])/Re/dz2+
                        (v[i+1][k][j]-2*v[i][k][j]+v[i+1][k][j])/Re/dphi2-
                        (sq(0.5*(v[i][k][j]+v[i][k+1][j]))-sq(0.5*(v[i][k-1][j]+v[i][k][j])))/dz-

                        0.25*((u[i][k][j]+  u[i][k+1][j])*  (v[i][k][j+1]+v[i][k][j])-
                              (u[i][k][j-1]+u[i][k+1][j-1])*(v[i][k][j]  +v[i][k][j-1])
                            )/dr-

                        0.25*((w[i]  [k][j]+w[i]  [k+1][j])*(v[i]  [k][j]+v[i+1][k][j])-
                              (w[i-1][k][j]+w[i-1][k+1][j])*(v[i-1][k][j]+v[i]  [k][j])
                            )/dphi);
                }
            }
        }
        // H
        for (int i = 0; i <= nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    H[i][k][j] = w[i][k][j] + dt*(
                        (w[i][k][j+1]-2*w[i][k][j]+w[i][k][j-1])/Re/dr2+
                        (w[i][k+1][j]-2*w[i][k][j]+w[i][k-1][j])/Re/dz2+
                        (w[i+1][k][j]-2*w[i][k][j]+w[i+1][k][j])/Re/dphi2-
                        (sq(0.5*(w[i+1][k][j]+w[i][k][j]))-sq(0.5*(w[i-1][k][j]+w[i][k][j])))/dphi-

                        0.25*((u[i+1][k][j]+  u[i][k][j])*  (w[i][k][j+1]+w[i][k][j])-
                              (u[i+1][k][j-1]+u[i][k][j-1])*(w[i][k][j]  +w[i][k][j-1])
                            )/dr-

                        0.25*((w[i][k][j]+  w[i][k+1][j])*(v[i][k]  [j]+v[i+1][k]  [j])-
                              (w[i][k-1][j]+w[i][k]  [j])*(v[i][k-1][j]+v[i+1][k-1][j])
                            )/dz);
                }
            }
        }
    }

    void init_P() {
        if (P_initialized) {
            return;
        }

        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    int id = RHS.index({i,k,j});
                    double r = i*dr-dr/2;
                    double r2 = r*r;

                    P.add(id, RHS.index(i-1,k,j), 1/dphi2/r2);

                    if (k > 1) {
                        P.add(id, RHS.index(i,k-1,j), 1/dz2);
                    }

                    if (j > 1) {
                        P.add(id, RHS.index(i,k,j-1), (r-0.5*dr)/dr2/r);
                    }

                    P.add(id, RHS.index(i,k,j), -2/dr2-2/dz2-2/dphi2/r2);

                    if (j < nr) {
                        P.add(id, RHS.index(i,k,j+1, (r+0.5*dr)/dr2/r));
                    }

                    if (k < nz) {
                        P.add(id, RHS.index(i,k+1,j), 1/dz2);
                    }

                    P.add(id, RHS.index(i+1,k,j), 1/dphi2/r2);
                }
            }
        }
        P.close();

        solver = std::move(P);

        P_initialized = true;
    }

    void poisson() {
        for (int i = 1; i <= nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    RHS[i][k][j] = ((F[i][k][j]-F[i][k][j-1])/dr
                                    +(G[i][k][j]-G[i][k-1][j])/dz
                                    +(H[i][k][j]-H[i-1][k][j])/dphi)/dt;

                    if (i <= 1) {
                        RHS[i][k][j] -= p[i-1][k][j]/dphi2;
                    }
                    if (k <= 1) {
                        RHS[i][k][j] -= p[i][k-1][j]/dz2;
                    }
                    if (j <= 1) {
                        RHS[i][k][j] -= p[i][k][j-1]/dr2;
                    }


                    if (j >= nr) {
                        RHS[i][k][j] -= p[i][k][j+1]/dr2;
                    }
                    if (k >= nz) {
                        RHS[i][k][j] -= p[i][k+1][j]/dz2;
                    }
                    if (i >= nphi) {
                        RHS[i][k][j] -= p[i+1][k][j]/dphi2;
                    }
                }
            }
        }

        solver.solve(&x[1][1][1], &RHS[1][1][1]);
    }

    void update_uvwp() {
        for (int i = 1; i <= nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j < nr; j++) {
                    u[i][k][j] = F[i][k][j]-dt/dr*(x[i][k][j+1]-x[i][k][j]);
                }
            }
        }

        for (int i = 1; i <= nphi; i++) {
            for (int k = 1; k < nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    v[i][k][j] = G[i][k][j]-dt/dz*(x[i][k+1][j]-x[i][k][j]);
                }
            }
        }

        for (int i = 1; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    w[i][k][j] = H[i][k][j]-dt/dphi*(x[i+1][k][j]-x[i][k][j]);
                }
            }
        }

        // TODO: check
        for (int i = 1; i < nphi; i++) {
            for (int k = 1; k < nz; k++) {
                for (int j = 1; j < nr; j++) {
                    p[i][k][j] = x[i][k][j];
                }
            }
        }
    }

    void update_uvi() {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                ui[k][j] = 0.5*(u[0][k][j-1] + u[0][k][j]);
                vi[k][j] = 0.5*(v[0][k-1][j] + v[0][k][j]);
            }
        }
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
