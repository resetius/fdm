#include <string>
#include <vector>
#include <climits>
#include <cmath>

#include "matrix.h"
#include "matrix_plot.h"
#include "config.h"
#include "sparse.h"

using namespace std;
using namespace fdm;

inline double sq(double x) {
    return x*x;
}

// Флетчер, том 2, страница 398
int main() {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);

    double x1 = c.get("ns", "x1", -M_PI);
    double y1 = c.get("ns", "y1", -M_PI);

    double x2 = c.get("ns", "x2", M_PI);
    double y2 = c.get("ns", "y2", M_PI);

    // число "ячеек"
    int nx = c.get("ns", "nx", 32);
    int ny = c.get("ns", "ny", 32);

    double Re = c.get("ns", "Re", 1.0);
    double dt = c.get("ns", "dt", 0.001);

    double dx = (x2-x1)/nx;
    double dy = (y2-y1)/ny;
    double dx2 = dx*dx;
    double dy2 = dy*dy;

    int np [[maybe_unused]] = nx*ny; // количество внутренних точек давления (p)
    int nu [[maybe_unused]] = (nx-1)*ny; // количество внутренних точек скорости (u)
    int nv [[maybe_unused]] = nx*(ny-1); // количество внутренних точек скорости (v)
    matrix<double> u(-1, 0, nx+1, ny+1), unext(-1, 0, nx+1, ny+1);
    matrix<double> v(0, -1, ny+1, nx+1), vnext(0, -1, ny+1, nx+1);
    matrix<double> p(ny+2, nx+2), pnext(ny+2, nx+2);

    matrix<double> F(0, 1, nx, ny), G(1, 0, nx, ny);

    vector<double> RHS(np);

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

    // TODO: boundary conditions

    // начальное условие
    for (int k = 0; k < ny+2; k++) {
        for (int j = -1; j <= nx+1; j++) {
            double y = y1+dy*k+dy/2;
            u[k][j] = -sin(y/2-M_PI/2);
        }
    }
    // свободная стенка
    for (int k = 0; k < ny+2; k++) {
        verify(fabs(u[k][-1] - u[k][1]) < 1e-14);
        verify(fabs(u[k][nx+1] - u[k][nx-1]) < 1e-14);
        u[k][-1] = u[k][1];
        u[k][nx+1] = u[k][nx-1];
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

    csr_matrix<double> P;

    P.clear();

#define pId(j,k) ((k-1)*nx+j-1)
    // 17.3
    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            // (j,k) -> row number
            int id = pId(j,k);
            RHS[id] = ((F[k][j]-F[k][j-1])/dx+(G[k][j]-G[k-1][j])/dy)/dt;

            if (k > 1) {
                P.add(id, pId(j,k-1), 1/dy2);
            } else {
                RHS[id] -= p[k-1][j]/dy2;
            }

            if (j > 1) {
                P.add(id, pId(j-1,k), 1/dx2);
            } else {
                RHS[id] -= p[k][j-1]/dx2;
            }

            P.add(id, pId(j,k), -2/dx2-2/dy2);

            if (j < nx) {
                P.add(id, pId(j+1,k), 1/dx2);
            } else {
                RHS[id] -= p[k][j+1]/dx2;
            }

            if (k < ny) {
                P.add(id, pId(j,k+1), 1/dy2);
            } else {
                RHS[id] -= p[k+1][j]/dy2;
            }
        }
    }
    P.close();
    //P.print();

    matrix<double> x(ny, nx);
    umfpack_solver<double> solver(std::move(P));
    solver.solve(&x[0][0], &RHS[0]);

    //for (int k = 0; k < ny+2; k++) {
    //for (int j = 0; j < nx+1; j++) {
    for (int k = 1; k <= ny; k++) {
    for (int j = 1; j <= nx; j++) {
        //printf("%0.1e ", x[k-1][j-1]);
            //printf("%f ", u[k][j]);
    }
    //printf("\n");
    }
    //printf("\n");

    matrix_plotter plotter;
    plotter.plot(matrix_plotter::settings(x)
                 .levels(10)
                 .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2)
                 .devname("pngcairo")
                 .fname("1.png"));


#undef pId
    return 0;
}
