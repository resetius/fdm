#pragma once

#include <string>
#include <vector>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdio>

#include "tensor.h"
#include "matrix_plot.h"
#include "config.h"
#include "lapl_cube.h"
#include "lapl_rect.h"

using namespace std;
using namespace fdm;

using asp::format;
using asp::sq;

namespace fdm {

template<typename T, bool check>
class NSCube {
public:
    using tensor = fdm::tensor<T,3,check>;
    using matrix = fdm::tensor<T,2,check>;

    const double x1,y1,z1;
    const double x2,y2,z2;
    const double U0; // скорость движения крышки

    const double Re;
    const double dt;

    const int nx, ny, nz;
    const double dx, dy, dz;
    const double dx2, dy2, dz2;

    tensor u /*x*/,v/*y*/,w/*z*/;
    tensor p,x;
    tensor F,G,H,RHS;
    matrix RHS_x,RHS_y,RHS_z;

    matrix psi_x;
    matrix psi_y;
    matrix psi_z; // срез по плоскости Oxy
    matrix uz, vz; // срез по плоскости Oxy
    matrix uy, wy; // срез по плоскости Oxz
    matrix vx, wx; // срез по плоскости Oyz

    LaplCube<T,check> lapl_solver;

    // для визуализации срезов
    LaplRect<T,check> lapl_x_solver;
    LaplRect<T,check> lapl_y_solver;
    LaplRect<T,check> lapl_z_solver;

    int time_index = 0;
    int plot_time_index = -1;

    NSCube(const Config& c)
        : x1(c.get("ns", "x1", -M_PI))
        , y1(c.get("ns", "y1", -M_PI))
        , z1(c.get("ns", "z1", -M_PI))
        , x2(c.get("ns", "x2",  M_PI))
        , y2(c.get("ns", "y2",  M_PI))
        , z2(c.get("ns", "z2",  M_PI))
        , U0(c.get("ns", "u0", 1.0))
        , Re(c.get("ns", "Re", 1.0))
        , dt(c.get("ns", "dt", 0.001))

        , nx(c.get("ns", "nx", 32))
        , ny(c.get("ns", "nx", 32))
        , nz(c.get("ns", "nz", 32))

        , dx((x2-x1)/nx), dy((y2-y1)/ny), dz((z2-z1)/nz)
        , dx2(dx*dx), dy2(dy*dy), dz2(dz*dz)

        , u{{0,  nz+1,  0, ny+1, -1, nx+1}}
        , v{{0,  nz+1, -1, ny+1,  0, nx+1}}
        , w{{-1, nz+1,  0, ny+1,  0, nx+1}}
        , p({0,  nz+1,  0, ny+1,  0, nx+1})

        , x({1, nz, 1, ny, 1, nx})
        , F({1, nz, 1, ny, 0, nx})
        , G({1, nz, 0, ny, 1, nx})
        , H({0, nz, 1, ny, 1, nx})
        , RHS({1, nz, 1, ny, 1, nx})

        , RHS_x({1, nz, 1, ny})
        , RHS_y({1, nz, 1, nx})
        , RHS_z({1, ny, 1, nx})

        , psi_x({1, nz, 1, ny})
        , psi_y({1, nz, 1, nx})
        , psi_z({1, ny, 1, nx})

        , uz({0, ny+1, 0, nx+1}) // inner u
        , vz({0, ny+1, 0, nx+1}) // inner v

        , uy({0, nz+1, 0, nx+1})
        , wy({0, nz+1, 0, nx+1})

        , vx({0, nz+1, 0, ny+1})
        , wx({0, nz+1, 0, ny+1})

        , lapl_solver(dx, dy, dz, x2-x1+dx, y2-y1+dy, z2-z1+dz, nx, ny, nz)

        , lapl_x_solver(dy, dz, y2-y1+dy, z2-z1+dz, ny, nz)
        , lapl_y_solver(dx, dz, x2-x1+dx, z2-z1+dz, nx, nz)
        , lapl_z_solver(dx, dy, x2-x1+dx, y2-y1+dy, nx, ny)
    { }

    void step();

    void plot();

    void vtk_out();

private:
    void init_bound();

    void FGH();

    void poisson();

    void poisson_stream();

    void update_stream() {
        poisson_stream();
    }

    void update_uvwp();

    void update_uvi();
};

} // namespace fdm
