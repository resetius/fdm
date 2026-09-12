#pragma once

#include <cmath>

#include "config.h"
#include "ns_cyl_sycl.h"

namespace fdm {

// Adapts the shared-USM fields of NSCylSycl to the field[i][k][j] interface
// consumed by NSCylStateLayout and the nonlinear experiment drivers.
template<typename T>
class NSCylSyclTask {
    struct Field {
        CylAcc<T> access;
        T* vec = nullptr;
        int size = 0;

        struct Row {
            CylAcc<T> access;
            int i;
            int k;
            T& operator[](int j) const { return access(i, k, j); }
        };
        struct Plane {
            CylAcc<T> access;
            int i;
            Row operator[](int k) const { return {access, i, k}; }
        };
        Plane operator[](int i) const { return {access, i}; }
    };

    static Field field(
        CylAcc<T> access, int nphi, int nz, int radial_size) {
        return {access, access.ptr, nphi*nz*radial_size};
    }

public:
    NSCylSyclTask(sycl::queue& queue, const Config& config)
        : nr(config.get("ns", "nr", 32))
        , nz(config.get("ns", "nz", 31))
        , nphi(config.get("ns", "nphi", 32))
        , r0(config.get("ns", "r", M_PI/2))
        , R(config.get("ns", "R", M_PI))
        , h1(config.get("ns", "h1", 0.0))
        , h2(config.get("ns", "h2", 10.0))
        , dr((R-r0)/nr)
        , dz((h2-h1)/nz)
        , dphi(2*M_PI/nphi)
        , dt(config.get("ns", "dt", 0.001))
        , Re(config.get("ns", "Re", 1.0))
        , U0(config.get("ns", "u0", 1.0))
        , queue_(queue)
        , ns_(queue, nr, nz, nphi, r0, R, h2-h1, U0, Re, dt)
        , u(field(ns_.ua(), nphi, nz, nr+3))
        , v(field(ns_.va(), nphi, nz, nr+2))
        , w(field(ns_.wa(), nphi, nz, nr+2))
        , p(field(ns_.pa(), nphi, nz, nr+2))
    { }

    void step() {
        ns_.step();
        ++time_index;
    }

    void wait() {
        queue_.wait_and_throw();
    }

    void apply_boundary_conditions() {
        ns_.apply_boundary_conditions();
        queue_.wait_and_throw();
    }

    CylAcc<T> ua() const { return u.access; }
    CylAcc<T> va() const { return v.access; }
    CylAcc<T> wa() const { return w.access; }
    CylAcc<T> pa() const { return p.access; }

    const int nr;
    const int nz;
    const int nphi;
    const T r0;
    const T R;
    const T h1;
    const T h2;
    const T dr;
    const T dz;
    const T dphi;
    const T dt;
    const T Re;
    const T U0;
    int time_index = 0;

private:
    sycl::queue& queue_;
    NSCylSycl<T> ns_;

public:
    Field u;
    Field v;
    Field w;
    Field p;
};

} // namespace fdm
