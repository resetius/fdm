#include <string>
#include <vector>

#include "config.h"

using namespace std;

inline double sq(double x) {
    return x*x;
}

template<typename T>
class matrix: public std::vector<T> {
    std::vector<T> vec;
    int rows;
    int cols;
    int rs;

public:
    matrix(int rows, int cols, int rs = 0): rows(rows), cols(cols), rs(rs?rs:cols)
    {
        vec.resize(rows*rs);
    }

    double& operator()(int y, int x) {
        return vec[y*rs+x];
    }

    double* operator[](int y) {
        return &vec[y*rs];
    }
};

// Флетчер, том 2, страница 398
int main() {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);

    double x1 = c.get("ns", "x1", -1.0);
    double y1 = c.get("ns", "y1", -1.0);

    double x2 = c.get("ns", "x2", 1.0);
    double y2 = c.get("ns", "y2", 1.0);

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
    matrix<double> u(ny, nx-1), unext(ny, nx-1);
    matrix<double> v(ny-1, nx), vnext(ny-1, nx);
    matrix<double> p(ny, nx), pnext(ny, nx);

    matrix<double> F(ny, nx-1), G(ny-1, nx);

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

    // F
    for (int j = 0; j < nx-1; j++) {
        for (int k = 0; k < ny; k++) {
            // TODO: boundary conditions
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
    for (int j = 0; j < nx; j++) {
        for (int k = 0; k < ny-1; k++) {
            // TODO: boundary conditions
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

    return 0;
}
