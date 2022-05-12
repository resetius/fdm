#include <string>
#include <vector>
#include <climits>
#include <cmath>

#include "matrix.h"
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

    int np = nx*ny; // количество внутренних точек давления (p)
    int nu = (nx-1)*ny; // количество внутренних точек скорости (u)
    int nv = nx*(ny-1); // количество внутренних точек скорости (v)
    matrix<double> u(ny+2, nx+1), unext(ny+2, nx+1);
    matrix<double> v(ny+1, nx+2), vnext(ny+1, nx+2);
    matrix<double> p(ny+2, nx+2), pnext(ny+2, nx+2);

    matrix<double> F(ny+2, nx+1), G(ny+1, nx+2);

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
    for (int k = 0; k < ny+2; k++) {
        for (int j = 0; j < nx+1; j++) {
            double y = y1+dy*k+dy/2;
            u[k][j] = -sin(y/2-M_PI/2);
        }
    }

    // F
    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j < nx; j++) {
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
    for (int k = 1; k < ny; k++) {
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
            // TODO: F[0][k], G[j][0] ...
            RHS[id] = ((F[j][k]-F[j-1][k])/dx+(G[j][k]-G[j][k-1])/dy)/dt;

            if (k > 1) {
                P.add(id, pId(j,k-1), 1/dy2);
            }
            // TODO: bnd to RHS

            if (j > 1) {
                P.add(id, pId(j-1,k), 1/dx2);
            }

            P.add(id, pId(j,k), -2/dx2-2/dy2);

            if (j < nx) {
                P.add(id, pId(j+1,k), 1/dx2);
            }

            if (k < ny) {
                P.add(id, pId(j,k+1), 1/dy2);
            }
        }
    }
    P.close();

    vector<double> x(np);
    umfpack_solver<double> solver(std::move(P));
    solver.solve(&x[0], &RHS[0]);

    //for (int k = 0; k < ny+2; k++) {
    //for (int j = 0; j < nx+1; j++) {
    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            printf("%f ", x[pId(j,k)]);
            //printf("%f ", u[k][j]);
        }
        printf("\n");
    }
    printf("\n");
#undef pId
    return 0;
}
