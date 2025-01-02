#include "ns_cube.h"
#include "unixbench_score.h"

#include <chrono>

using namespace std;
using namespace fdm;

using asp::format;
using asp::sq;

namespace fdm {

template<typename T, bool check>
NSCube<T,check>::~NSCube()
{
    if (verbose) {
        printf("Step times (ms): init_bound=%.2f, FGH=%.2f, poisson=%.2f, update_uvwp=%.2f\n",
                unixbench_score(init_bound_times),
                unixbench_score(fgh_times),
                unixbench_score(poisson_times),
                unixbench_score(update_times));
    }
}

template<typename T, bool check>
void NSCube<T,check>::step() {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time;
    double init_bound_time, fgh_time, poisson_time, update_time;

    init_bound();
    end_time = std::chrono::high_resolution_clock::now();
    init_bound_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    start_time = end_time;

    FGH();
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
               dt*time_index, p.maxabs(), u.maxabs(), v.maxabs(), w.maxabs(), x.maxabs(),
               RHS.maxabs(), F.maxabs(), G.maxabs(), H.maxabs());

        init_bound_times.emplace_back(init_bound_time);
        fgh_times.emplace_back(fgh_time);
        poisson_times.emplace_back(poisson_time);
        update_times.emplace_back(update_time);
    }
}

template<typename T, bool check>
void NSCube<T,check>::init_bound() {
    // крышка
    for (int k = 0; k <= ny+1; k++) {
        for (int j = -1; j <= nz+1; j++) {
            // 0.5*(u[nz+1][k][j] + u[nz][k][j]) = U0
            u[nz+1][k][j] = 2*U0 - u[nz][k][j];
        }
    }

    // div=0 на границе
    // инициализация узлов за пределами области
    for (int i = 0; i <= nz+1; i++) {
        for (int k = 0; k <= ny+1; k++) {
            u[i][k][-1]   = u[i][k][1];
            u[i][k][nx+1] = u[i][k][nx-1];
        }
    }

    for (int i = 0; i <= nz+1; i++) {
        for (int j = 0; j <= nx+1; j++) {
            v[i][-1][j]   = v[i][1][j];
            v[i][ny+1][j] = v[i][ny-1][j];
        }
    }

    for (int k = 0; k <= ny+1; k++) {
        for (int j = 0; j <= nx+1; j++) {
            w[-1][k][j]   = w[1][k][j];
            w[nz+1][k][j] = w[nz-1][k][j];
        }
    }

    // давление
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            p[i][k][0]  = p[i][k][1] -
                (u[i][k][1]-2*u[i][k][0]+u[i][k][-1])/Re/dx;
            p[i][k][nx+1] = p[i][k][nx]-
                (u[i][k][nx+1]-2*u[i][k][nx]+u[i][k][nx-1])/Re/dx;
        }
    }
    for (int i = 1; i <= nz; i++) {
        for (int j = 1; j <= nx; j++) {
            p[i][0][j]    = p[i][1][j] -
                (v[i][1][j]-2*v[i][0][j]+v[i][-1][j])/Re/dy;
            p[i][ny+1][j] = p[i][ny][j]-
                (v[i][ny+1][j]-2*v[i][ny][j]+v[i][ny-1][j])/Re/dy;
        }
    }
    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            p[0][k][j]    = p[1][k][j] -
                (w[1][k][j]-2*w[0][k][j]+w[-1][k][j])/Re/dz;
            p[nz+1][k][j] = p[nz][k][j]-
                (w[nz+1][k][j]-2*w[nz][k][j]+w[nz-1][k][j])/Re/dz;
        }
    }
}


template<typename T, bool check>
void NSCube<T,check>::FGH() {
#pragma omp parallel
    { // omp parallel

    // F (r)
#pragma omp for collapse(2) schedule(static)
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) { // 3/2 ..
            for (int j = 0; j <= nx; j++) { // 1/2 ..
                // 17.9
                F[i][k][j] = u[i][k][j] + dt*(
                    (   u[i][k][j+1]-2*u[i][k][j]+   u[i][k][j-1])/Re/dx2+
                    (   u[i][k+1][j]-2*u[i][k][j]+   u[i][k-1][j])/Re/dy2+
                    (   u[i+1][k][j]-2*u[i][k][j]+   u[i-1][k][j])/Re/dz2-
                    (   sq(0.5*(u[i][k][j]+u[i][k][j+1]))-sq(0.5*(u[i][k][j-1]+u[i][k][j])))/dx-

                    0.25*((u[i][k]  [j]+u[i][k+1][j])*(v[i][k]  [j+1]+v[i][k]  [j])-
                          (u[i][k-1][j]+u[i][k]  [j])*(v[i][k-1][j+1]+v[i][k-1][j])
                        )/dy-

                    0.25*((u[i]  [k][j]+u[i+1][k][j])*(w[i]  [k][j+1]+w[i]  [k][j])-
                          (u[i-1][k][j]+u[i]  [k][j])*(w[i-1][k][j+1]+w[i-1][k][j])
                        )/dz
                    );
            }
        }
    }
    // G (z)
#pragma omp for collapse(2) schedule(static)
    for (int i = 1; i <= nz; i++) {
        for (int k = 0; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                // 17.11
                G[i][k][j] = v[i][k][j] + dt*(
                    (   v[i][k][j+1]-2*v[i][k][j]+   v[i][k][j-1])/Re/dx2+
                    (   v[i][k+1][j]-2*v[i][k][j]+   v[i][k-1][j])/Re/dy2+
                    (   v[i+1][k][j]-2*v[i][k][j]+   v[i-1][k][j])/Re/dz2-
                    (sq(0.5*(v[i][k][j]+v[i][k+1][j]))-sq(0.5*(v[i][k-1][j]+v[i][k][j])))/dy-

                    0.25*((u[i][k][j]+  u[i][k+1][j])*  (v[i][k][j+1]+v[i][k][j])-
                          (u[i][k][j-1]+u[i][k+1][j-1])*(v[i][k][j]  +v[i][k][j-1])
                        )/dx-

                    0.25*((w[i]  [k][j]+w[i]  [k+1][j])*(v[i]  [k][j]+v[i+1][k][j])-
                          (w[i-1][k][j]+w[i-1][k+1][j])*(v[i-1][k][j]+v[i]  [k][j])
                        )/dz
                    );
            }
        }
    }
    // H (phi)
#pragma omp for collapse(2) schedule(static)
    for (int i = 0; i <= nz; i++) { // 1/2 ...
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                H[i][k][j] = w[i][k][j] + dt*(
                    (   w[i][k][j+1]-2*w[i][k][j]+   w[i][k][j-1])/Re/dx2+
                    (   w[i][k+1][j]-2*w[i][k][j]+   w[i][k-1][j])/Re/dy2+
                    (   w[i+1][k][j]-2*w[i][k][j]+   w[i-1][k][j])/Re/dz2-
                    (sq(0.5*(w[i+1][k][j]+w[i][k][j]))-sq(0.5*(w[i-1][k][j]+w[i][k][j])))/dz-

                    0.25*((u[i+1][k][j]+  u[i][k][j])*  (w[i][k][j+1]+w[i][k][j])-
                          (u[i+1][k][j-1]+u[i][k][j-1])*(w[i][k][j]  +w[i][k][j-1])
                        )/dx-

                    0.25*((w[i][k][j]+  w[i][k+1][j])*(v[i][k]  [j]+v[i+1][k]  [j])-
                          (w[i][k-1][j]+w[i][k]  [j])*(v[i][k-1][j]+v[i+1][k-1][j])
                        )/dy
                    );
            }
        }
    }

    } // end of omp parallel
}


template<typename T, bool check>
void NSCube<T,check>::poisson() {
#pragma omp parallel for collapse(2)
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                RHS[i][k][j] =  ((F[i][k][j]-F[i][k][j-1])/dx
                                 +(G[i][k][j]-G[i][k-1][j])/dy
                                 +(H[i][k][j]-H[i-1][k][j])/dz)/dt;

                if (i <= 1) {
                    RHS[i][k][j] -= p[i-1][k][j]/dz2;
                }
                if (k <= 1) {
                    RHS[i][k][j] -= p[i][k-1][j]/dy2;
                }
                if (j <= 1) {
                    RHS[i][k][j] -= p[i][k][j-1]/dx2;
                }


                if (j >= nx) {
                    RHS[i][k][j] -= p[i][k][j+1]/dx2;
                }
                if (k >= ny) {
                    RHS[i][k][j] -= p[i][k+1][j]/dy2;
                }
                if (i >= nz) {
                    RHS[i][k][j] -= p[i+1][k][j]/dz2;
                }
            }
        }
    }

    lapl_solver.solve(&x[1][1][1], &RHS[1][1][1]);
}

template<typename T, bool check>
void NSCube<T,check>::update_uvwp() {
#pragma omp parallel
    { // omp parallel

#pragma omp for collapse(2) schedule(static)
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j < nx; j++) {
                //double r = r0+dr*j;
                u[i][k][j] = F[i][k][j]-dt/dx*(x[i][k][j+1]-x[i][k][j]);
            }
        }
    }
#pragma omp for collapse(2) schedule(static)
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k < ny; k++) {
            for (int j = 1; j <= nx; j++) {
                //double r = r0+dr*j-dr/2;
                v[i][k][j] = G[i][k][j]-dt/dy*(x[i][k+1][j]-x[i][k][j]);
            }
        }
    }
#pragma omp for collapse(2) schedule(static)
    for (int i = 1; i < nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                w[i][k][j] = H[i][k][j]-dt/dz*(x[i+1][k][j]-x[i][k][j]);
            }
        }
    }

    } // end of omp parallel

    {
        p = x;
    }
}

template class NSCube<double,true>;
template class NSCube<double,false>;
template class NSCube<float,true>;
template class NSCube<float,false>;

} // namespace fdm
