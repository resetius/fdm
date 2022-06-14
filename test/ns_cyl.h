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
#include "sparse.h"
#include "asp_misc.h"
#include "lapl_cyl.h"

using namespace std;
using namespace fdm;

using asp::format;
using asp::sq;

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

    csr_matrix<T> P_r, P_z, P_phi;

    LaplCyl3FFT2<T,check,zflag> lapl3_solver;

    int time_index = 0;
    int plot_time_index = -1;

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

    void step() {
        init_bound();
        FGH();
        poisson();
        update_uvwp();
        time_index++;
        if (verbose) {
            printf("%.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e \n",
                   dt*time_index,
                   p.maxabs(), u.maxabs(), v.maxabs(), w.maxabs(), x.maxabs(),
                   RHS.maxabs(), F.maxabs(), G.maxabs(), H.maxabs());
        }
    }

    void L_step() {
        init_bound();
        L_FGH();
        poisson();
        update_uvwp();
        time_index++;
        if (verbose) {
            printf("%.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e \n",
                   dt*time_index,
                   p.maxabs(), u.maxabs(), v.maxabs(), w.maxabs(), x.maxabs(),
                   RHS.maxabs(), F.maxabs(), G.maxabs(), H.maxabs());
        }
    }

private:
    void init_bound() {
        // внутренний цилиндр
        for (int i = 0; i < nphi; i++) {
            for (int k = z0; k <= znn; k++) {
                // 0.5*(w[i][k][0] + w[i][k][1]) = U0
                w[i][k][0] = 2*U0 - w[i][k][1]; // inner
                w[i][k][nr+1] = -w[i][k][nr]; // outer
            }
        }
        for (int i = 0; i < nphi; i++) {
            for (int k = z_; k <= znn; k++) {
                v[i][k][0]    = - v[i][k][1]; // inner
                v[i][k][nr+1] = -v[i][k][nr]; // outer
            }
        }

        // div=0 на границе
        // инициализация узлов за пределами области
        for (int i = 0; i < nphi; i++) {
            for (int k = z0; k <= znn; k++) {
                u[i][k][-1]   = u[i][k][1];
                u[i][k][nr+1] = u[i][k][nr-1];
            }
        }

        if constexpr(zflag==tensor_flag::none) {
            for (int i = 0; i < nphi; i++) {
                for (int j = z0; j <= znn; j++) {
                    v[i][-1][j]   = v[i][1][j];
                    v[i][nz+1][j] = v[i][nz-1][j];
                }
            }

            // check me
            for (int i = 0; i < nphi; i++) {
                for (int j = -1; j <= nr+1; j++) {
                    u[i][0][j]    = -u[i][1][j];
                    u[i][nz+1][j] = -u[i][nz][j];
                }
            }

            for (int i = 0; i < nphi; i++) {
                for (int j = 0; j <= nr+1; j++) {
                    w[i][0][j]    = -w[i][1][j];
                    w[i][nz+1][j] = -w[i][nz][j];
                }
            }
        }

        // давление
        // check me
        for (int i = 0; i < nphi; i++) {
            for (int k = z1; k <= zn; k++) {
                int j = 0;
                double r = r0+j*dr-dr/2;
                verify(std::abs(u[i][k][j]) < 1e-15);
                verify(std::abs(u[i][k][j-1]-u[i][k][j+1])<1e-15);

                //p[i][k][0]  = (r+0.5*dr)/(r-0.5*dr) * p[i][k][1] -
                //    ((r+0.5*dr)*u[i][k][1]/r-2*u[i][k][0]+(r-0.5*dr)*u[i][k][-1]/r)/Re/dr;
                p[i][k][0]  = p[i][k][1] -
                    ((r+0.5*dr)*u[i][k][1]/r-2*u[i][k][0]+(r-0.5*dr)*u[i][k][-1]/r)/Re/dr;

                j = nr;
                r = r0+j*dr-dr/2;

                verify(std::abs(u[i][k][j]) < 1e-15);
                verify(std::abs(u[i][k][j-1]-u[i][k][j+1])<1e-15);

                //p[i][k][nr+1] = (r-0.5*dr)/(r+0.5*dr) * p[i][k][nr]+
                //    ((r+0.5*dr)*u[i][k][nr+1]/r-2*u[i][k][nr]+(r-0.5*dr)*u[i][k][nr-1]/r)/Re/dr;
                p[i][k][nr+1] = p[i][k][nr]+
                    ((r+0.5*dr)*u[i][k][nr+1]/r-2*u[i][k][nr]+(r-0.5*dr)*u[i][k][nr-1]/r)/Re/dr;
            }
        }
        if constexpr(zflag==tensor_flag::none) {
            for (int i = 0; i < nphi; i++) {
                for (int j = 1; j <= nr; j++) {
                    verify(std::abs(v[i][0][j]) < 1e-15);
                    verify(std::abs(v[i][nz][j]) < 1e-15);

                    verify(std::abs(v[i][1][j]-v[i][-1][j]) < 1e-15);
                    verify(std::abs(v[i][nz+1][j]-v[i][nz-1][j]) < 1e-15);

                    p[i][0][j]    = p[i][1][j] -
                        (v[i][1][j]-2*v[i][0][j]+v[i][-1][j])/Re/dz;
                    p[i][nz+1][j] = p[i][nz][j]+
                        (v[i][nz+1][j]-2*v[i][nz][j]+v[i][nz-1][j])/Re/dz;
                }
            }
        }
    }

    void FGH() {
#pragma omp parallel
        { // omp parallel

#pragma omp single
        { // omp single

        // F (r)
#pragma omp task
        for (int i = 1; i <= nphi; i++) {
            for (int k = z1; k <= zn; k++) { // 3/2 ..
                for (int j = 0; j <= nr; j++) { // 1/2 ..
                    double r = r0+dr*j;
                    double r2 = (r+0.5*dr)/r;
                    double r1 = (r-0.5*dr)/r;
                    double rr = r*r;

                    // 17.9
                    F[i][k][j] = u[i][k][j] + dt*(
                        (r2*u[i][k][j+1]-2*u[i][k][j]+r1*u[i][k][j-1])/Re/dr2+
                        (   u[i][k+1][j]-2*u[i][k][j]+   u[i][k-1][j])/Re/dz2+
                        (   u[i+1][k][j]-2*u[i][k][j]+   u[i-1][k][j])/Re/dphi2/rr-
                        (r2*sq(0.5*(u[i][k][j]+u[i][k][j+1]))-r1*sq(0.5*(u[i][k][j-1]+u[i][k][j])))/dr-

                        0.25*((u[i][k]  [j]+u[i][k+1][j])*(v[i][k]  [j+1]+v[i][k]  [j])-
                              (u[i][k-1][j]+u[i][k]  [j])*(v[i][k-1][j+1]+v[i][k-1][j])
                            )/dz-

                        0.25*((u[i]  [k][j]+u[i+1][k][j])*(w[i]  [k][j+1]+w[i]  [k][j])-
                              (u[i-1][k][j]+u[i]  [k][j])*(w[i-1][k][j+1]+w[i-1][k][j])
                            )/dphi/r

                        // TODO: check
                        +sq(0.5*(w[i][k][j+1]+w[i][k][j]))/r-u[i][k][j]/rr/Re
                        -2*( 0.5*(w[i]  [k][j+1]+w[i]  [k][j])
                            -0.5*(w[i-1][k][j+1]+w[i-1][k][j]))/rr/dphi/Re
                        );
                }
            }
        }
        // G (z)
#pragma omp task
        for (int i = 0; i < nphi; i++) {
            for (int k = z0; k <= zn; k++) {
                for (int j = 1; j <= nr; j++) {
                    double r = r0+dr*j-dr/2;
                    double r2 = (r+0.5*dr)/r;
                    double r1 = (r-0.5*dr)/r;
                    double rr = r*r;

                    // 17.11
                    G[i][k][j] = v[i][k][j] + dt*(
                        (r2*v[i][k][j+1]-2*v[i][k][j]+r1*v[i][k][j-1])/Re/dr2+
                        (   v[i][k+1][j]-2*v[i][k][j]+   v[i][k-1][j])/Re/dz2+
                        (   v[i+1][k][j]-2*v[i][k][j]+   v[i-1][k][j])/Re/dphi2/rr-
                        (sq(0.5*(v[i][k][j]+v[i][k+1][j]))-sq(0.5*(v[i][k-1][j]+v[i][k][j])))/dz-

                        // TODO: check
                        0.25*(r2*(u[i][k][j]+  u[i][k+1][j])*  (v[i][k][j+1]+v[i][k][j])-
                              r1*(u[i][k][j-1]+u[i][k+1][j-1])*(v[i][k][j]  +v[i][k][j-1])
                            )/dr-

                        0.25*((w[i]  [k][j]+w[i]  [k+1][j])*(v[i]  [k][j]+v[i+1][k][j])-
                              (w[i-1][k][j]+w[i-1][k+1][j])*(v[i-1][k][j]+v[i]  [k][j])
                            )/dphi/r
                        );
                }
            }
        }
        // H (phi)
#pragma omp task
        for (int i = 0; i < nphi; i++) { // 1/2 ...
            for (int k = z1; k <= zn; k++) {
                for (int j = 1; j <= nr; j++) {
                    double r = r0+dr*j-dr/2;
                    double r2 = (r+0.5*dr)/r;
                    double r1 = (r-0.5*dr)/r;
                    double rr = r*r;

                    H[i][k][j] = w[i][k][j] + dt*(
                        (r2*w[i][k][j+1]-2*w[i][k][j]+r1*w[i][k][j-1])/Re/dr2+
                        (   w[i][k+1][j]-2*w[i][k][j]+   w[i][k-1][j])/Re/dz2+
                        (   w[i+1][k][j]-2*w[i][k][j]+   w[i-1][k][j])/Re/dphi2/rr-
                        (sq(0.5*(w[i+1][k][j]+w[i][k][j]))-sq(0.5*(w[i-1][k][j]+w[i][k][j])))/dphi/r-

                        // TODO: check
                        0.25*(r2*(u[i+1][k][j]+  u[i][k][j])*  (w[i][k][j+1]+w[i][k][j])-
                              r1*(u[i+1][k][j-1]+u[i][k][j-1])*(w[i][k][j]  +w[i][k][j-1])
                            )/dr-

                        0.25*((w[i][k][j]+  w[i][k+1][j])*(v[i][k]  [j]+v[i+1][k]  [j])-
                              (w[i][k-1][j]+w[i][k]  [j])*(v[i][k-1][j]+v[i+1][k-1][j])
                            )/dz

                        // TODO: check

                        -w[i][k][j]*0.5*(u[i+1][k][j]+u[i][k][j])/r-w[i][k][j]/rr/Re
                        +2*( 0.5*(u[i+1][k][j]+u[i]  [k][j])
                           -0.5*(u[i]  [k][j]+u[i-1][k][j]))/rr/dphi/Re
                        );
                }
            }
        }

#pragma omp taskwait

        } // end of omp single
        } // end of omp parallel
    }

    void L_FGH() {
#pragma omp parallel
        { // omp parallel

#pragma omp single
        { // omp single

        // F (r)
#pragma omp task
        for (int i = 1; i <= nphi; i++) {
            for (int k = z1; k <= zn; k++) { // 3/2 ..
                for (int j = 0; j <= nr; j++) { // 1/2 ..
                    double r = r0+dr*j;
                    double r2 = (r+0.5*dr)/r;
                    double r1 = (r-0.5*dr)/r;
                    double rr = r*r;

                    // 17.9
                    F[i][k][j] = u[i][k][j] + dt*(
                        (r2*u[i][k][j+1]-2*u[i][k][j]+r1*u[i][k][j-1])/Re/dr2+
                        (   u[i][k+1][j]-2*u[i][k][j]+   u[i][k-1][j])/Re/dz2+
                        (   u[i+1][k][j]-2*u[i][k][j]+   u[i-1][k][j])/Re/dphi2/rr-

                        (r2*(0.5*u[i][k][j]+u[i][k][j+1])*(u0[i][k][j]+u0[i][k][j+1])
                        -r1*(0.5*u[i][k][j-1]+u[i][k][j])*(u0[i][k][j-1]+u0[i][k][j]))/dr-

                        0.25*((u[i][k]  [j]+u[i][k+1][j])*(v0[i][k]  [j+1]+v0[i][k]  [j])-
                              (u[i][k-1][j]+u[i][k]  [j])*(v0[i][k-1][j+1]+v0[i][k-1][j])
                            )/dz-
                        0.25*((u0[i][k]  [j]+u0[i][k+1][j])*(v[i][k]  [j+1]+v[i][k]  [j])-
                              (u0[i][k-1][j]+u0[i][k]  [j])*(v[i][k-1][j+1]+v[i][k-1][j])
                            )/dz-

                        0.25*((u[i]  [k][j]+u[i+1][k][j])*(w0[i]  [k][j+1]+w0[i]  [k][j])-
                              (u[i-1][k][j]+u[i]  [k][j])*(w0[i-1][k][j+1]+w0[i-1][k][j])
                            )/dphi/r-
                        0.25*((u0[i]  [k][j]+u0[i+1][k][j])*(w[i]  [k][j+1]+w[i]  [k][j])-
                              (u0[i-1][k][j]+u0[i]  [k][j])*(w[i-1][k][j+1]+w[i-1][k][j])
                            )/dphi/r

                        +0.5*(w[i][k][j+1]+w[i][k][j])*(w0[i][k][j+1]+w0[i][k][j])/r
                        -u[i][k][j]/rr/Re
                        -2*( 0.5*(w[i]  [k][j+1]+w[i]  [k][j])
                            -0.5*(w[i-1][k][j+1]+w[i-1][k][j]))/rr/dphi/Re
                        );
                }
            }
        }
        // G (z)
#pragma omp task
        for (int i = 0; i < nphi; i++) {
            for (int k = z0; k <= zn; k++) {
                for (int j = 1; j <= nr; j++) {
                    double r = r0+dr*j-dr/2;
                    double r2 = (r+0.5*dr)/r;
                    double r1 = (r-0.5*dr)/r;
                    double rr = r*r;

                    // 17.11
                    G[i][k][j] = v[i][k][j] + dt*(
                        (r2*v[i][k][j+1]-2*v[i][k][j]+r1*v[i][k][j-1])/Re/dr2+
                        (   v[i][k+1][j]-2*v[i][k][j]+   v[i][k-1][j])/Re/dz2+
                        (   v[i+1][k][j]-2*v[i][k][j]+   v[i-1][k][j])/Re/dphi2/rr-

                        (0.5*(v[i][k][j]+v[i][k+1][j])*(v0[i][k][j]+v0[i][k+1][j])
                        -0.5*(v[i][k-1][j]+v[i][k][j])*(v0[i][k-1][j]+v0[i][k][j]))/dz-

                        0.25*(r2*(u[i][k][j]+ u[i][k+1][j])* (v0[i][k][j+1]+v0[i][k][j])-
                            r1*(u[i][k][j-1]+u[i][k+1][j-1])*(v0[i][k][j]  +v0[i][k][j-1])
                            )/dr-
                        0.25*(r2*(u0[i][k][j]+u0[i][k+1][j])*  (v[i][k][j+1]+v[i][k][j])-
                            r1*(u0[i][k][j-1]+u0[i][k+1][j-1])*(v[i][k][j]  +v[i][k][j-1])
                            )/dr-

                        0.25*((w[i]  [k][j]+w[i]  [k+1][j])*(v0[i]  [k][j]+v0[i+1][k][j])-
                              (w[i-1][k][j]+w[i-1][k+1][j])*(v0[i-1][k][j]+v0[i]  [k][j])
                            )/dphi/r-
                        0.25*((w0[i]  [k][j]+w0[i]  [k+1][j])*(v[i]  [k][j]+v[i+1][k][j])-
                              (w0[i-1][k][j]+w0[i-1][k+1][j])*(v[i-1][k][j]+v[i]  [k][j])
                            )/dphi/r
                        );
                }
            }
        }
        // H (phi)
#pragma omp task
        for (int i = 0; i < nphi; i++) { // 1/2 ...
            for (int k = z1; k <= zn; k++) {
                for (int j = 1; j <= nr; j++) {
                    double r = r0+dr*j-dr/2;
                    double r2 = (r+0.5*dr)/r;
                    double r1 = (r-0.5*dr)/r;
                    double rr = r*r;

                    H[i][k][j] = w[i][k][j] + dt*(
                        (r2*w[i][k][j+1]-2*w[i][k][j]+r1*w[i][k][j-1])/Re/dr2+
                        (   w[i][k+1][j]-2*w[i][k][j]+   w[i][k-1][j])/Re/dz2+
                        (   w[i+1][k][j]-2*w[i][k][j]+   w[i-1][k][j])/Re/dphi2/rr-

                        (0.5*(w[i+1][k][j]+w[i][k][j])*(w0[i+1][k][j]+w0[i][k][j])
                       -0.5*(w[i-1][k][j]+w[i][k][j])*(w0[i-1][k][j]+w0[i][k][j]))/dphi/r-

                        0.25*(r2*(u[i+1][k][j]+  u[i][k][j])* (w0[i][k][j+1]+w0[i][k][j])-
                              r1*(u[i+1][k][j-1]+u[i][k][j-1])*(w0[i][k][j]+w0[i][k][j-1])
                            )/dr-
                        0.25*(r2*(u0[i+1][k][j]+u0[i][k][j])*  (w[i][k][j+1]+w[i][k][j])-
                            r1*(u0[i+1][k][j-1]+u0[i][k][j-1])*(w[i][k][j]  +w[i][k][j-1])
                            )/dr-

                        0.25*((w[i][k][j]+  w[i][k+1][j])*(v0[i][k]  [j]+v0[i+1][k]  [j])-
                              (w[i][k-1][j]+w[i][k]  [j])*(v0[i][k-1][j]+v0[i+1][k-1][j])
                            )/dz-
                        0.25*((w0[i][k][j]+  w0[i][k+1][j])*(v[i][k]  [j]+v[i+1][k]  [j])-
                              (w0[i][k-1][j]+w0[i][k]  [j])*(v[i][k-1][j]+v[i+1][k-1][j])
                            )/dz

                        -w0[i][k][j]*0.5*(u[i+1][k][j]+u[i][k][j])/r
                        -w[i][k][j]*0.5*(u0[i+1][k][j]+u0[i][k][j])/r

                        -w[i][k][j]/rr/Re
                        +2*( 0.5*(u[i+1][k][j]+u[i]  [k][j])
                           -0.5*(u[i]  [k][j]+u[i-1][k][j]))/rr/dphi/Re
                        );
                }
            }
        }

#pragma omp taskwait

        } // end of omp single
        } // end of omp parallel
    }

    void poisson() {
#pragma omp parallel for
        for (int i = 0; i < nphi; i++) {
            for (int k = z1; k <= zn; k++) {
                for (int j = 1; j <= nr; j++) {
                    double r = r0+dr*j-dr/2;

                    RHS[i][k][j] = (((r+0.5*dr)*F[i][k][j]-(r-0.5*dr)*F[i][k][j-1])/r/dr
                                    +(G[i][k][j]-G[i][k-1][j])/dz
                                    +(H[i][k][j]-H[i-1][k][j])/dphi/r)/dt;

                    if constexpr(zflag==tensor_flag::none) {
                        if (k <= 1) {
                            RHS[i][k][j] -= p[i][k-1][j]/dz2;
                        }
                    }
                    if (j <= 1) {
                        RHS[i][k][j] -= (r-dr/2)/r*p[i][k][j-1]/dr2;
                    }


                    if (j >= nr) {
                        RHS[i][k][j] -= (r+dr/2)/r*p[i][k][j+1]/dr2;
                    }
                    if constexpr(zflag==tensor_flag::none) {
                        if (k >= nz) {
                            RHS[i][k][j] -= p[i][k+1][j]/dz2;
                        }
                    }
                }
            }
        }

        lapl3_solver.solve(&x[0][z1][1], &RHS[0][z1][1]);
    }

    void update_uvwp() {
#pragma omp parallel
        { // omp parallel

#pragma omp single
        { // omp single

#pragma omp task
        for (int i = 0; i < nphi; i++) {
            for (int k = z1; k <= zn; k++) {
                for (int j = 1; j < nr; j++) {
                    //double r = r0+dr*j;
                    u[i][k][j] = F[i][k][j]-dt/dr*(x[i][k][j+1]-x[i][k][j]);
                }
            }
        }

#pragma omp task
        for (int i = 0; i < nphi; i++) {
            for (int k = z1; k < nz /*ok*/; k++) {
                for (int j = 1; j <= nr; j++) {
                    //double r = r0+dr*j-dr/2;
                    v[i][k][j] = G[i][k][j]-dt/dz*(x[i][k+1][j]-x[i][k][j]);
                }
            }
        }

#pragma omp task
        for (int i = 0; i < nphi; i++) {
            for (int k = z1; k <= zn; k++) {
                for (int j = 1; j <= nr; j++) {
                    double r = r0+dr*j-dr/2;
                    w[i][k][j] = H[i][k][j]-dt/dphi/r*(x[i+1][k][j]-x[i][k][j]);
                }
            }
        }

#pragma omp task
        {
            p = x;
        }

#pragma omp taskwait
        } // end of omp single
        } // end of omp parallel
    }
};
