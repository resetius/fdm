#include "ns_cyl.h"
#include "ns_cyl_fgh.h"
#include "unixbench_score.h"

using namespace std;

using asp::format;
using asp::sq;

namespace fdm {

template<typename T, bool check, tensor_flag zflag>
NSCyl<T,check,zflag>::~NSCyl() {
    if (verbose) {
        printf("Step times (ms): init_bound=%.2f, FGH=%.2f, poisson=%.2f, update_uvwp=%.2f\n",
                unixbench_score(init_bound_times),
                unixbench_score(fgh_times),
                unixbench_score(poisson_times),
                unixbench_score(update_times));
    }
}

template<typename T, bool check, tensor_flag zflag>
void NSCyl<T,check,zflag>::step() {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time;
    double init_bound_time, fgh_time, poisson_time, update_time;

    init_bound();
    end_time = std::chrono::high_resolution_clock::now();
    init_bound_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    start_time = end_time;

    FGH();
    apply_outer_boundary_step_data();
    end_time = std::chrono::high_resolution_clock::now();
    fgh_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    start_time = end_time;

    poisson();
    end_time = std::chrono::high_resolution_clock::now();
    poisson_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    start_time = end_time;

    update_uvwp();
    end_time = std::chrono::high_resolution_clock::now();
    update_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    time_index++;
    if (verbose) {
        printf("%.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e \n",
               dt*time_index,
               p.maxabs(), u.maxabs(), v.maxabs(), w.maxabs(), x.maxabs(),
               RHS.maxabs(), F.maxabs(), G.maxabs(), H.maxabs());
        if (verbose > 1) {
            printf("Step times (ms): init_bound=%.2f, FGH=%.2f, poisson=%.2f, update_uvwp=%.2f\n",
                   init_bound_time, fgh_time, poisson_time, update_time);
        }

        init_bound_times.emplace_back(init_bound_time);
        fgh_times.emplace_back(fgh_time);
        poisson_times.emplace_back(poisson_time);
        update_times.emplace_back(update_time);
    }
}

template<typename T, bool check, tensor_flag zflag>
void NSCyl<T,check,zflag>::L_step() {
    init_bound();
    L_FGH();
    apply_outer_boundary_step_data();
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

template<typename T, bool check, tensor_flag zflag>
void NSCyl<T,check,zflag>::init_bound() {
    // внутренний цилиндр
    for (int i = 0; i < nphi; i++) {
        for (int k = z0; k <= znn; k++) {
            // 0.5*(w[i][k][0] + w[i][k][1]) = U0
            w[i][k][0] = 2*U0 - w[i][k][1]; // inner
            const T outer = outer_azimuthal_velocity(i, k);
            // 0.5*(w[nr]+w[nr+1]) = outer wall velocity.
            w[i][k][nr+1] = 2*outer-w[i][k][nr];
        }
    }
    for (int i = 0; i < nphi; i++) {
        for (int k = z_; k <= znn; k++) {
            v[i][k][0]    = - v[i][k][1]; // inner
            const T outer = outer_axial_velocity(i, k);
            // 0.5*(v[nr]+v[nr+1]) = outer wall velocity.
            v[i][k][nr+1] = 2*outer-v[i][k][nr];
        }
    }

    // div=0 на границе
    // инициализация узлов за пределами области
    for (int i = 0; i < nphi; i++) {
        for (int k = z0; k <= znn; k++) {
            u[i][k][nr] = outer_radial_velocity(i, k);
            u[i][k][-1]   = u[i][k][1];
            u[i][k][nr+1] = u[i][k][nr-1];
        }
    }

    if constexpr(zflag==tensor_flag::none) {
        for (int i = 0; i < nphi; i++) {
            for (int j = 0; j <= nr+1; j++) {
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

    for (int i = 0; i < nphi; i++) {
        for (int k = z1; k <= zn; k++) {
            verify(std::abs(u[i][k][0]) < 1e-15);
            verify(std::abs(u[i][k][nr]
                            -outer_radial_velocity(i, k)) < 1e-15);
        }
    }
    if constexpr(zflag==tensor_flag::none) {
        for (int i = 0; i < nphi; i++) {
            for (int j = 1; j <= nr; j++) {
                verify(std::abs(v[i][0][j]) < 1e-15);
                verify(std::abs(v[i][nz][j]) < 1e-15);

                verify(std::abs(v[i][1][j]-v[i][-1][j]) < 1e-15);
                verify(std::abs(v[i][nz+1][j]-v[i][nz-1][j]) < 1e-15);
            }
        }
    }
}

template<typename T, bool check, tensor_flag zflag>
void NSCyl<T,check,zflag>::FGH() {
    const NSCylFGHParams<T> params(
        r0, dr, dz, dphi, dr2, dz2, dphi2, dt, Re);

    if constexpr(zflag == tensor_flag::periodic) {
#pragma omp parallel for collapse(2)
        for (int i = 0; i < nphi; ++i) {
            for (int k = 0; k < nz; ++k) {
                for (int face = 0; face <= nr; ++face) {
                    ns_cyl_fgh_node(
                        u, v, w, F, G, H,
                        i, k, face, nr, params);
                }
            }
        }
        return;
    }

#pragma omp parallel
    {

    // F (r)
#pragma omp for collapse(2)
    for (int i = 0; i < nphi; i++) {
        for (int k = z1; k <= zn; k++) { // 3/2 ..
            for (int j = 0; j <= nr; j++) { // 1/2 ..
                ns_cyl_f_node(u, v, w, F, i, k, j, params);
            }
        }
    }
    // G (z)
#pragma omp for collapse(2)
    for (int i = 0; i < nphi; i++) {
        for (int k = z0; k <= zn; k++) {
            for (int j = 1; j <= nr; j++) {
                ns_cyl_g_node(u, v, w, G, i, k, j, params);
            }
        }
    }
    // H (phi)
#pragma omp for collapse(2)
    for (int i = 0; i < nphi; i++) { // 1/2 ...
        for (int k = z1; k <= zn; k++) {
            for (int j = 1; j <= nr; j++) {
                ns_cyl_h_node(u, v, w, H, i, k, j, params);
            }
        }
    }

    } // end of omp parallel
}

template<typename T, bool check, tensor_flag zflag>
void NSCyl<T,check,zflag>::L_FGH() {
#pragma omp parallel
    { // omp parallel

    // F (r)
#pragma omp for collapse(2)
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

                    0.5*(r2*(u[i][k][j]+u[i][k][j+1])*(u0[i][k][j]+u0[i][k][j+1])
                         -r1*(u[i][k][j-1]+u[i][k][j])*(u0[i][k][j-1]+u0[i][k][j]))/dr-

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
#pragma omp for collapse(2)
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
#pragma omp for collapse(2)
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

    } // end of omp parallel
}

template<typename T, bool check, tensor_flag zflag>
void NSCyl<T,check,zflag>::apply_outer_boundary_step_data() {
    if (!outer_boundary_step_data_enabled_
        || outer_radial_predictor_.empty()) {
        return;
    }
    for (int i = 0; i < nphi; ++i) {
        for (int k = z1; k <= zn; ++k) {
            F[i][k][nr] =
                outer_radial_predictor_[outer_boundary_index(i, k)];
        }
    }
}

template<typename T, bool check, tensor_flag zflag>
void NSCyl<T,check,zflag>::poisson() {
    // Radial pressure is Neumann data at the new time level.  Its unknown
    // interior value is part of the first/last matrix diagonal; only the
    // known flux contribution is added to RHS below.
    if constexpr(zflag==tensor_flag::none) {
        for (int i = 0; i < nphi; i++) {
            for (int j = 1; j <= nr; j++) {
                p[i][0][j] = p[i][1][j] - dz*G[i][0][j]/dt;
                p[i][nz+1][j] = p[i][nz][j] + dz*G[i][nz][j]/dt;
            }
        }
    }

#pragma omp parallel for collapse(2)
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
                    RHS[i][k][j] += (r-dr/2)/r
                        *F[i][k][0]/(dr*dt);
                }


                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r
                        *(F[i][k][nr]-outer_radial_velocity_next(i, k))
                        /(dr*dt);
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

template<typename T, bool check, tensor_flag zflag>
void NSCyl<T,check,zflag>::update_uvwp() {
#pragma omp parallel
    { // omp parallel

#pragma omp for collapse(2)
    for (int i = 0; i < nphi; i++) {
        for (int k = z1; k <= zn; k++) {
            for (int j = 1; j < nr; j++) {
                //double r = r0+dr*j;
                u[i][k][j] = F[i][k][j]-dt/dr*(x[i][k][j+1]-x[i][k][j]);
            }
        }
    }

#pragma omp for collapse(2)
    for (int i = 0; i < nphi; i++) {
        for (int k = z1; k < nz /*ok*/; k++) {
            for (int j = 1; j <= nr; j++) {
                //double r = r0+dr*j-dr/2;
                v[i][k][j] = G[i][k][j]-dt/dz*(x[i][k+1][j]-x[i][k][j]);
            }
        }
    }

#pragma omp for collapse(2)
    for (int i = 0; i < nphi; i++) {
        for (int k = z1; k <= zn; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;
                w[i][k][j] = H[i][k][j]-dt/dphi/r*(x[i+1][k][j]-x[i][k][j]);
            }
        }
    }

    } // end of omp parallel

    {
        p = x;
    }

    for (int i = 0; i < nphi; ++i) {
        for (int k = z1; k <= zn; ++k) {
            p[i][k][0] = p[i][k][1]-dr*F[i][k][0]/dt;
            p[i][k][nr+1] = p[i][k][nr]
                +dr*(F[i][k][nr]-outer_radial_velocity_next(i, k))/dt;
        }
    }

    if (outer_boundary_step_data_enabled_) {
        for (int i = 0; i < nphi; ++i) {
            for (int k = z1; k <= zn; ++k) {
                u[i][k][nr] = outer_radial_velocity_next(i, k);
            }
        }
        outer_radial_velocity_.swap(outer_radial_velocity_next_);
        if (!outer_axial_velocity_next_.empty()) {
            outer_axial_velocity_.swap(outer_axial_velocity_next_);
            outer_azimuthal_velocity_.swap(
                outer_azimuthal_velocity_next_);
        }
        outer_radial_predictor_.clear();
        outer_radial_velocity_next_.clear();
        outer_axial_velocity_next_.clear();
        outer_azimuthal_velocity_next_.clear();
        outer_boundary_step_data_enabled_ = false;
    }
}

template class NSCyl<double,true,tensor_flag::none>;
template class NSCyl<double,false,tensor_flag::none>;
template class NSCyl<float,true,tensor_flag::none>;
template class NSCyl<float,false,tensor_flag::none>;

template class NSCyl<double,true,tensor_flag::periodic>;
template class NSCyl<double,false,tensor_flag::periodic>;
template class NSCyl<float,true,tensor_flag::periodic>;
template class NSCyl<float,false,tensor_flag::periodic>;

} // namespace fdm
