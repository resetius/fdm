#pragma once

#include <string>
#include <vector>
#include <climits>
#include <cmath>
#include <chrono>
#include <random>

#include "tensor.h"
#include "matrix_plot.h"
#include "config.h"
#include "asp_misc.h"
#include "lapl_cyl.h"

namespace fdm {

template<typename T, bool check, tensor_flag zflag=tensor_flag::none>
class NSCyl {
public:
    using tensor_flags = typename fdm::short_flags<tensor_flag::periodic, zflag> :: value;
    using tensor = fdm::tensor<T,3,check,tensor_flags>;

    const double R, r0;
    const double h1, h2;
    double U0; // скорость вращения внутреннего цилиндра

    const double Re;
    const double dt;

    const int nr, nz, nphi;
    const int verbose;

    const int z_, z0, z1, zn, znn; // z bounds

    const double dr, dz, dphi;
    const double dr2, dz2, dphi2;

    tensor u /*r*/,v/*z*/,w/*phi*/, p;
    // linearization near this point
    tensor u0, v0, w0;

    tensor x;
    tensor F,G,H,RHS;

    LaplCyl3FFT2<T,check,zflag> lapl3_solver;

    int time_index = 0;

    NSCyl(const Config& c)
        : R(c.get("ns", "R", M_PI))
        , r0(c.get("ns", "r", M_PI/2))
        , h1(c.get("ns", "h1", 0))
        , h2(c.get("ns", "h2", 10))
        , U0(c.get("ns", "u0", 1.0))
        , Re(c.get("ns", "Re", 1.0))
        , dt(c.get("ns", "dt", 0.001))

        , nr(c.get("ns", "nr", 32))
        , nz(c.get("ns", "nz", 31))
        , nphi(c.get("ns", "nphi", 32))
        , verbose(c.get("ns", "verbose", 0))

        , z_(zflag==tensor_flag::none?-1:0)
        , z0(zflag==tensor_flag::none?0:0)
        , z1(zflag==tensor_flag::none?1:0)
        , zn(zflag==tensor_flag::none?nz:nz-1)
        , znn(zflag==tensor_flag::none?nz+1:nz-1)

        , dr((R-r0)/nr), dz((h2-h1)/nz), dphi(2*M_PI/nphi)
        , dr2(dr*dr), dz2(dz*dz), dphi2(dphi*dphi)

          // phi, z, r
        , u{{0, nphi-1, z0, znn, -1, nr+1}}
        , v{{0, nphi-1, z_, znn, 0, nr+1}}
        , w{{0, nphi-1, z0, znn, 0, nr+1}}
        , p({0, nphi-1, z0, znn, 0, nr+1})

        , u0{{0, nphi-1, z0, znn, -1, nr+1}}
        , v0{{0, nphi-1, z_, znn, 0, nr+1}}
        , w0{{0, nphi-1, z0, znn, 0, nr+1}}

        , x({0, nphi-1, z1, zn, 1, nr})
        , F({0, nphi-1, z1, zn, 0, nr}) // check bounds
        , G({0, nphi-1, z0, zn, 1, nr}) // check bounds
        , H({0, nphi-1, z1, zn, 1, nr}) // check bounds
        , RHS({0, nphi-1, z1, zn, 1, nr})

        , lapl3_solver(dr, dz, r0-dr/2, R-r0+dr,
                       zflag==tensor_flag::none?h2-h1+dz:h2-h1,
                       nr, nz, nphi)
    {
        if (c.get("ns", "vrandom", 0) == 1) {
            std::default_random_engine generator;
            std::uniform_real_distribution<T> distribution(-1e-3, 1e-3);
            for (int i = 0; i < nphi; i++) {
                for (int k = z1; k <= zn; k++) {
                    for (int j = 1; j <= nr; j++) {
                        v[i][k][j] = distribution(generator);
                    }
                }
            }
        }
    }

    int size() const {
        return u.size+v.size+w.size+p.size;
    }

    void step();

    void L_step();

private:
    void init_bound();

    void FGH();

    void L_FGH();

    void poisson();

    void update_uvwp();
};

} // namespace fdm
