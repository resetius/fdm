#pragma once

#include <string>
#include <vector>
#include <climits>
#include <cmath>
#include <chrono>
#include <random>
#include <stdexcept>

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
    double U0; // азимутальная скорость поверхности внутреннего цилиндра

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

    // debug
    std::vector<double> init_bound_times;
    std::vector<double> fgh_times;
    std::vector<double> poisson_times;
    std::vector<double> update_times;

    NSCyl(const Config& c)
        : R(c.get("ns", "R", M_PI))
        , r0(c.get("ns", "r", M_PI/2))
        , h1(c.get("ns", "h1", 0.0))
        , h2(c.get("ns", "h2", 10.0))
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
                       nr, nz, nphi,
                       lapl_cyl_radial_boundary::neumann)
    {
        if (c.get("ns", "vrandom", 0) == 1) {
            std::default_random_engine generator;
            std::uniform_real_distribution<T> distribution(-1e-3, 1e-3);
            for (int i = 0; i < nphi; i++) {
                // v задана на z-гранях; k=0,nz — стенки непериодического цилиндра.
                for (int k = z1; k < nz; k++) {
                    for (int j = 1; j <= nr; j++) {
                        v[i][k][j] = distribution(generator);
                    }
                }
            }
        }
    }

    ~NSCyl();

    int size() const {
        return u.size+v.size+w.size+p.size;
    }

    // Азимутальная скорость Куэтта при неподвижном внешнем цилиндре.
    double couette_velocity(double r) const {
        const double denominator = R*R-r0*r0;
        const double A = -U0*r0/denominator;
        const double B = U0*r0*R*R/denominator;
        return A*r+B/r;
    }

    void step();

    void L_step();

    void apply_boundary_conditions() {
        init_bound();
    }

    // Prescribe the velocity of the outer cylindrical wall on the periodic
    // (phi,z) grid.  The radial component lives directly on r=R; axial and
    // azimuthal components are imposed through the cell-centred ghost values.
    // Keeping this state separate from u/v/w is important: every step rebuilds
    // the ghost layer, while the prescribed wall value must persist.
    void set_outer_boundary_velocity(
        const std::vector<T>& radial,
        const std::vector<T>& axial,
        const std::vector<T>& azimuthal) {
        validate_outer_boundary_plane(radial, axial, azimuthal);
        outer_radial_velocity_ = radial;
        outer_axial_velocity_ = axial;
        outer_azimuthal_velocity_ = azimuthal;
        outer_boundary_velocity_enabled_ = true;
        outer_boundary_step_data_enabled_ = false;
        outer_radial_predictor_.clear();
        outer_radial_velocity_next_.clear();
        outer_axial_velocity_next_.clear();
        outer_azimuthal_velocity_next_.clear();
    }

    void set_outer_boundary_velocity(
        const std::vector<T>& axial,
        const std::vector<T>& azimuthal) {
        set_outer_boundary_velocity(
            std::vector<T>(static_cast<std::size_t>(nphi)*nz, T(0)),
            axial, azimuthal);
    }

    // Supply the data of an artificial radial interface for one time step.
    // The normal predictor is evaluated by the enclosing-domain stencil,
    // and radial_next is the projected interface velocity after the step.
    // The pressure trace is not prescribed: the same-time-level Neumann
    // condition follows from radial_predictor and radial_next.
    // After the step radial_next becomes the ordinary prescribed velocity;
    // callers advancing a moving interface must supply fresh data each step.
    void set_outer_boundary_step_data(
        const std::vector<T>& radial,
        const std::vector<T>& axial,
        const std::vector<T>& azimuthal,
        const std::vector<T>& radial_predictor,
        const std::vector<T>& radial_next) {
        validate_outer_boundary_plane(radial, axial, azimuthal);
        const std::size_t expected =
            static_cast<std::size_t>(nphi)*nz;
        if (radial_predictor.size() != expected
            || radial_next.size() != expected) {
            throw std::invalid_argument(
                "outer boundary step data has the wrong plane size");
        }
        outer_radial_velocity_ = radial;
        outer_axial_velocity_ = axial;
        outer_azimuthal_velocity_ = azimuthal;
        outer_radial_predictor_ = radial_predictor;
        outer_radial_velocity_next_ = radial_next;
        outer_axial_velocity_next_.clear();
        outer_azimuthal_velocity_next_.clear();
        outer_boundary_velocity_enabled_ = true;
        outer_boundary_step_data_enabled_ = true;
    }

    // Prescribe a time-dependent physical wall over one complete step.  F at
    // the wall is evaluated by this domain's momentum stencil; radial_next
    // supplies the new-time normal flux required by the pressure Neumann
    // condition.  All three next values become the persistent wall velocity
    // after the step.
    void set_outer_boundary_step_data(
        const std::vector<T>& radial,
        const std::vector<T>& axial,
        const std::vector<T>& azimuthal,
        const std::vector<T>& radial_next,
        const std::vector<T>& axial_next,
        const std::vector<T>& azimuthal_next) {
        validate_outer_boundary_plane(radial, axial, azimuthal);
        validate_outer_boundary_plane(
            radial_next, axial_next, azimuthal_next);
        outer_radial_velocity_ = radial;
        outer_axial_velocity_ = axial;
        outer_azimuthal_velocity_ = azimuthal;
        outer_radial_predictor_.clear();
        outer_radial_velocity_next_ = radial_next;
        outer_axial_velocity_next_ = axial_next;
        outer_azimuthal_velocity_next_ = azimuthal_next;
        outer_boundary_velocity_enabled_ = true;
        outer_boundary_step_data_enabled_ = true;
    }

    void clear_outer_boundary_velocity() {
        outer_boundary_velocity_enabled_ = false;
        outer_radial_velocity_.clear();
        outer_axial_velocity_.clear();
        outer_azimuthal_velocity_.clear();
        outer_boundary_step_data_enabled_ = false;
        outer_radial_predictor_.clear();
        outer_radial_velocity_next_.clear();
        outer_axial_velocity_next_.clear();
        outer_azimuthal_velocity_next_.clear();
    }

    bool has_outer_boundary_velocity() const {
        return outer_boundary_velocity_enabled_;
    }

private:
    bool outer_boundary_velocity_enabled_ = false;
    bool outer_boundary_step_data_enabled_ = false;
    std::vector<T> outer_radial_velocity_;
    std::vector<T> outer_axial_velocity_;
    std::vector<T> outer_azimuthal_velocity_;
    std::vector<T> outer_radial_predictor_;
    std::vector<T> outer_radial_velocity_next_;
    std::vector<T> outer_axial_velocity_next_;
    std::vector<T> outer_azimuthal_velocity_next_;

    void validate_outer_boundary_plane(
        const std::vector<T>& radial,
        const std::vector<T>& axial,
        const std::vector<T>& azimuthal) const {
        if constexpr(zflag != tensor_flag::periodic) {
            throw std::invalid_argument(
                "spatially varying outer boundary requires periodic z");
        }
        const std::size_t expected =
            static_cast<std::size_t>(nphi)*nz;
        if (radial.size() != expected || axial.size() != expected
            || azimuthal.size() != expected) {
            throw std::invalid_argument(
                "outer boundary velocity has the wrong plane size");
        }
    }

    std::size_t outer_boundary_index(int i, int k) const {
        return static_cast<std::size_t>(i)*nz+k;
    }

    T outer_radial_velocity(int i, int k) const {
        return outer_boundary_velocity_enabled_
            ? outer_radial_velocity_[outer_boundary_index(i, k)] : T(0);
    }

    T outer_radial_velocity_next(int i, int k) const {
        return outer_boundary_step_data_enabled_
            ? outer_radial_velocity_next_[outer_boundary_index(i, k)]
            : outer_radial_velocity(i, k);
    }

    T outer_axial_velocity(int i, int k) const {
        return outer_boundary_velocity_enabled_
            ? outer_axial_velocity_[outer_boundary_index(i, k)] : T(0);
    }

    T outer_azimuthal_velocity(int i, int k) const {
        return outer_boundary_velocity_enabled_
            ? outer_azimuthal_velocity_[outer_boundary_index(i, k)] : T(0);
    }

    void init_bound();

    void FGH();

    void L_FGH();

    void apply_outer_boundary_step_data();

    void poisson();

    void update_uvwp();
};

} // namespace fdm
