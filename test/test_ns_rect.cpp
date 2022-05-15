#include <string>
#include <vector>
#include <climits>
#include <cmath>
#include <cstdarg>

#include "matrix.h"
#include "matrix_plot.h"
#include "config.h"
#include "sparse.h"

using namespace std;
using namespace fdm;

inline double sq(double x) {
    return x*x;
}

char _format_buf[32768];

static std::string format(const char* format, ...) {
    va_list args;
    va_start (args, format);
    vsnprintf(_format_buf, sizeof(_format_buf), format, args);
    va_end(args);
    return _format_buf;
}

// Флетчер, том 2, страница 398
template<typename T, bool check>
class NSRect {
public:
    using matrix = fdm::matrix<T,check>;
    const double x1, y1;
    const double x2, y2;
    // число "ячеек"
    const int nx, ny;

    const double Re;
    const double dt;

    const double dx, dy;
    const double dx2, dy2;

    matrix u,v,p,x;
    matrix ui, vi;

    matrix F,G,RHS;

    matrix psi;

    csr_matrix<T> P;
    bool P_initialized = false;

    umfpack_solver<T> solver;
    umfpack_solver<T> stream_solver; // for stream function
    int time_index = 0;

    NSRect(const Config& c)
        : x1(c.get("ns", "x1", -M_PI))
        , y1(c.get("ns", "y1", -M_PI))
        , x2(c.get("ns", "x2", M_PI))
        , y2(c.get("ns", "y2", M_PI))
        , nx(c.get("ns", "nx", 32))
        , ny(c.get("ns", "ny", 32))
        , Re(c.get("ns", "Re", 1.0))
        , dt(c.get("ns", "dt", 0.001))
        , dx((x2-x1)/nx), dy((y2-y1)/ny)
        , dx2(dx*dx), dy2(dy*dy)

        , u(-1, 0, nx+1, ny+1)
        , v(0, -1, nx+1, ny+1)
        , p(0, 0, nx+1, ny+1)
        , x(1, 1, nx, ny) // inner p
        , ui(1, 1, nx, ny) // inner u
        , vi(1, 1, nx, ny) // inner v

        , F(0, 1, nx, ny)
        , G(1, 0, nx, ny)
        , RHS(1, 1, nx, ny)
        , psi(1, 1, nx, ny)
    {
        // начальное условие
        for (int k = 0; k < ny+2; k++) {
            for (int j = -1; j <= 0 /*nx+1*/; j++) {
                double y = y1+dy*k+dy/2;
                u[k][j] = -sin(y/2-M_PI/2);
            }
        }

        init_P();
    }

    void step()
    {
        init_bound();
        FG();
        poisson();
        update_uvp();
        ++time_index;
    }

    void plot()
    {
        update_uvi();
        update_stream();

        matrix_plotter plotter(matrix_plotter::settings()
                               .sub(2, 2)
                               .devname("pngcairo")
                               .fname(format("step_%03d.png", time_index)));
        plotter.plot(matrix_plotter::page()
                     .scalar(ui)
                     .levels(10)
                     .tlabel(format("U (%.1e)", dt*time_index))
                     .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));
        plotter.plot(matrix_plotter::page()
                     .scalar(vi)
                     .levels(10)
                     .tlabel(format("V (%.1e)", dt*time_index))
                     .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));
        plotter.plot(matrix_plotter::page()
                     .scalar(p)
                     .levels(10)
                     .tlabel(format("P (%.1e)", dt*time_index))
                     .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));
        plotter.plot(matrix_plotter::page()
                     .scalar(psi)
                     .levels(10)
                     .tlabel(format("Psi (%.1e)", dt*time_index))
                     .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));
//        plotter.plot(matrix_plotter::page()
//                     .vector(ui, vi)
//                     .levels(10)
//                     .tlabel(format("UV (%.1e)", dt*time_index))
//                     .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));
    }

    void plot_extra()
    {
        matrix_plotter plotter(matrix_plotter::settings()
                               .sub(2, 1)
                               .devname("pngcairo")
                               .fname(format("step_extra_%03d.png", time_index)));
        plotter.plot(matrix_plotter::page(F)
                     .scalar(F)
                     .levels(10)
                     .tlabel(format("F (%.1e)", dt*time_index))
                     .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));
        plotter.plot(matrix_plotter::page()
                     .scalar(G)
                     .levels(10)
                     .tlabel(format("G (%.1e)", dt*time_index))
                     .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));
    }

private:
    // TODO: check boundary conditions
    void init_bound() {
        // свободная стенка
        for (int k = 0; k < ny+2; k++) {
            //verify(time_index>0||fabs(u[k][-1] - u[k][1]) < 1e-14);
            //verify(time_index>0||fabs(u[k][nx+1] - u[k][nx-1]) < 1e-14);
            //u[k][-1] = u[k][1];
            //u[k][nx+1] = u[k][nx-1];
            // выход
            u[k][nx+1] = u[k][nx] = u[k][nx-1];
        }
        for (int k = -1; k <= ny+1; k++) {
            // вход
            v[k][0] = -v[k][1];
            // выход
            v[k][nx+1] = v[k][nx];
        }
        // твердая стенка
        for (int j = -1; j <= nx+1; j++) {
            // низ
            // u[1/2][j] = 0 <- стенка
            // u[1/2][j] = 0 = 0.5 (u[1][j] + u[0][j])
            // verify(fabs(u[0][j] + u[1][j]) < 1e-14);
            u[0][j] = -u[1][j];
            // верх
            u[ny+1][j] = -u[ny][j];
        }
        // давление
        for (int k = 0; k <= ny+1; k++) {
            // лево
            p[k][0] = p[k][1];
            // право
            p[k][nx+1] = p[k][nx];
        }
        for (int j = 0; j <= nx+1; j++) {
            // низ
            p[0][j] = p[1][j];
            // верх
            p[ny+1][j] = p[ny][j];
        }
    }

    void FG() {
        // F
        for (int k = 1; k <= ny; k++) { // 3/2 ..
            for (int j = 0; j <= nx; j++) { // 1/2 ..
                // 17.9
                F[k][j] = u[k][j] + dt*(
                    (u[k][j+1]-2*u[k][j]+u[k][j-1])/Re/dx2+
                    (u[k-1][j]-2*u[k][j]+u[k+1][j])/Re/dy2-
                    (sq(0.5*(u[k][j]+u[k][j+1]))-sq(0.5*(u[k][j-1]+u[k][j])))/dx-
                    0.25*((u[k]  [j]+u[k+1][j])*(v[k]  [j+1]+v[k]  [j])-
                          (u[k-1][j]+u[k]  [j])*(v[k-1][j+1]+v[k-1][j])
                        )/dy);
            }
        }
        // G
        for (int k = 0; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                // 17.11
                G[k][j] = v[k][j] + dt*(
                    (v[k][j+1]-2*v[k][j]+v[k][j-1])/Re/dx2+
                    (v[k+1][j]-2*v[k][j]+v[k-1][j])/Re/dy2-
                    (sq(0.5*(v[k][j]+v[k+1][j]))-sq(0.5*(v[k-1][j]+v[k][j])))/dy-
                    0.25*((u[k][j]+  u[k+1][j])*  (v[k][j+1]+v[k][j])-
                          (u[k][j-1]+u[k+1][j-1])*(v[k][j]  +v[k][j-1])
                        )/dx);
            }
        }
    }

    void init_P() {
        if (P_initialized) {
            return;
        }

        // 17.3
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                // (j,k) -> row number
                int id = RHS.index(k,j);

                if (k > 1) {
                    P.add(id, RHS.index(k-1,j), 1/dy2);
                }

                if (j > 1) {
                    P.add(id, RHS.index(k,j-1), 1/dx2);
                }

                P.add(id, RHS.index(k,j), -2/dx2-2/dy2);

                if (j < nx) {
                    P.add(id, RHS.index(k, j+1), 1/dx2);
                }

                if (k < ny) {
                    P.add(id, RHS.index(k+1,j), 1/dy2);
                }
            }
        }
        P.close();

        solver = std::move(P);

        P_initialized = true;
    }

    void poisson() {
        // 17.3
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                // (j,k) -> row number
                RHS[k][j] = ((F[k][j]-F[k][j-1])/dx+(G[k][j]-G[k-1][j])/dy)/dt;

                if (k > 1) {
                    ;
                } else {
                    RHS[k][j] -= p[k-1][j]/dy2;
                }

                if (j > 1) {
                    ;
                } else {
                    RHS[k][j] -= p[k][j-1]/dx2;
                }

                if (j < nx) {
                    ;
                } else {
                    RHS[k][j] -= p[k][j+1]/dx2;
                }

                if (k < ny) {
                    ;
                } else {
                    RHS[k][j] -= p[k+1][j]/dy2;
                }
            }
        }

        solver.solve(&x[1][1], &RHS[1][1]);
    }

    void update_stream() {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                RHS[k][j] = (v[k][j]-v[k][j-1])/dx-(u[k][j]-u[k-1][j])/dy;
                // zero conditions

                /*
                if (k <= 1)  { RHS[k][j] -= 1/dy2; }
                if (j <= 1)  { RHS[k][j] -= 1/dx2; }
                if (j >= nx) { RHS[k][j] -= 1/dx2; }
                if (k >= ny) { RHS[k][j] -= 1/dy2; }
                */
            }
        }

        solver.solve(&psi[1][1], &RHS[1][1]);
    }

    void update_uvp() {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j < nx; j++) {
                u[k][j] = F[k][j]-dt/dx*(x[k][j+1]-x[k][j]);
            }
        }

        for (int k = 1; k < ny; k++) {
            for (int j = 1; j <= nx; j++) {
                v[k][j] = G[k][j]-dt/dx*(x[k+1][j]-x[k][j]);
            }
        }

        p = x;
    }

    void update_uvi() {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                ui[k][j] = 0.5*(u[k][j-1] + u[k][j]);
                vi[k][j] = 0.5*(v[k-1][j] + v[k][j]);
            }
        }
    }
};

template<typename T, bool check>
void calc(const Config& c) {
    NSRect<T, true> ns(c);
    /*

            v=0,u=0
           ________
     p=0   |      | u/n=0
     u=u_0 |      | v/n=0
           |      |
           ________
            v=0,u=0

         \int Lapl p = 0 ?

     */

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
