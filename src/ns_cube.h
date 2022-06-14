#pragma once

#include <vector>

#include "tensor.h"
#include "matrix_plot.h"
#include "config.h"
#include "lapl_cube.h"
#include "lapl_rect.h"

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

    LaplCube<T,check> lapl_solver;

    int time_index = 0;

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

        , lapl_solver(dx, dy, dz, x2-x1+dx, y2-y1+dy, z2-z1+dz, nx, ny, nz)
    { }

    void step();

private:
    void init_bound();

    void FGH();

    void poisson();

    void update_uvwp();
};

} // namespace fdm
