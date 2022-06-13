#include "ns_cube.h"

using namespace std;
using namespace fdm;

using asp::format;
using asp::sq;

namespace fdm {

template<typename T, bool check>
void NSCube<T,check>::step() {
    init_bound();
    FGH();
    poisson();
    update_uvwp();
    time_index++;
    printf("%.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e \n",
           dt*time_index, p.maxabs(), u.maxabs(), v.maxabs(), w.maxabs(), x.maxabs(),
           RHS.maxabs(), F.maxabs(), G.maxabs(), H.maxabs());
}

template<typename T, bool check>
void NSCube<T,check>::plot() {
    update_uvi();
    update_stream();

    matrix_plotter plotter(matrix_plotter::settings()
                           .sub(2, 3)
                           .devname("pngcairo")
                           .fname(format("step_%07d.png", time_index)));

    plotter.plot(matrix_plotter::page()
                 .scalar(uz)
                 .levels(10)
                 .labels("Y", "X", "")
                 .tlabel(format("U (t=%.1e, |max|=%.1e)", dt*time_index, uz.maxabs()))
                 .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));
    plotter.plot(matrix_plotter::page()
                 .scalar(vz)
                 .levels(10)
                 .labels("Y", "X", "")
                 .tlabel(format("V (t=%.1e, |max|=%.1e)", dt*time_index, vz.maxabs()))
                 .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));
    plotter.plot(matrix_plotter::page()
                 .scalar(wy)
                 .levels(10)
                 .labels("Z", "X", "")
                 .tlabel(format("W (t=%.1e, |max|=%.1e)", dt*time_index, wy.maxabs()))
                 .bounds(x1+dx/2, z1+dz/2, x2-dx/2, z2-dz/2));
/*
  plotter.plot(matrix_plotter::page()
  .vector(uz, vz)
  .levels(10)
  .labels("Y", "X", "")
  .tlabel(format("UV (z=const) (%.1e)", dt*time_index))
  .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));
*/
    plotter.plot(matrix_plotter::page()
                 .scalar(psi_z)
                 .levels(10)
                 .labels("Y", "X", "")
                 .tlabel(format("UV (z=const) (%.1e)", dt*time_index))
                 .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));

/*
  plotter.plot(matrix_plotter::page()
  .vector(uy, wy)
  .levels(10)
  .labels("Z", "X", "")
  .tlabel(format("UW (y=const) (%.1e)", dt*time_index))
  .bounds(x1+dx/2, z1+dz/2, x2-dx/2, z2-dz/2));
*/

    plotter.plot(matrix_plotter::page()
                 .scalar(psi_y)
                 .levels(10)
                 .labels("Z", "X", "")
                 .tlabel(format("UW (y=const) (%.1e)", dt*time_index))
                 .bounds(x1+dx/2, z1+dz/2, x2-dx/2, z2-dz/2));

/*
  plotter.plot(matrix_plotter::page()
  .vector(vx, wx)
  .levels(10)
  .labels("Z", "Y", "")
  .tlabel(format("VW (x=const) (%.1e)", dt*time_index))
  .bounds(y1+dy/2, z1+dz/2, y2-dy/2, z2-dz/2));
*/
    plotter.plot(matrix_plotter::page()
                 .scalar(psi_x)
                 .levels(10)
                 .labels("Z", "Y", "")
                 .tlabel(format("VW (x=const) (%.1e)", dt*time_index))
                 .bounds(y1+dy/2, z1+dz/2, y2-dy/2, z2-dz/2));

}

template<typename T, bool check>
void NSCube<T,check>::vtk_out() {
    update_uvi();

    FILE* f = fopen(format("step_%07d.vtk", time_index).c_str(), "wb");
    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "step %d\n", time_index);
    fprintf(f, "ASCII\n");
    fprintf(f, "DATASET STRUCTURED_POINTS\n");
    fprintf(f, "DIMENSIONS %d %d %d\n", nx, ny, nz);
    fprintf(f, "ASPECT_RATIO 1 1 1\n");
    fprintf(f, "ORIGIN %f %f %f\n", x1+dx/2, y1+dy/2, z1+dz/2);
    fprintf(f, "SPACING %f %f %f\n", dx, dy, dz);
    fprintf(f, "POINT_DATA %d\n", nx*ny*nz);
    fprintf(f, "VECTORS u double\n");
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                fprintf(f, "%f %f %f\n",
                        0.5*(u[i][k][j]+u[i][k][j-1]),
                        0.5*(v[i][k][j]+v[i][k-1][j]),
                        0.5*(w[i][k][j]+w[i-1][k][j])
                    );
            }
        }
    }
    fclose(f);
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

#pragma omp single
    { // omp single

    // F (r)
#pragma omp task
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
#pragma omp task
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
#pragma omp task
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

#pragma omp taskwait

    } // end of omp single
    } // end of omp parallel
}


template<typename T, bool check>
void NSCube<T,check>::poisson() {
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
void NSCube<T,check>::poisson_stream() {
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            RHS_x[i][k] = (wx[i][k+1]-wx[i][k-1]) /2/ dy - (vx[i+1][k] - vx[i-1][k]) /2/ dz;
        }
    }

    lapl_x_solver.solve(&psi_x[1][1], &RHS_x[1][1]);


    for (int i = 1; i <= nz; i++) {
        for (int j = 1; j <= nx; j++) {
            RHS_y[i][j] = (wy[i][j+1]-wy[i][j-1]) /2/ dx - (uy[i+1][j] - uy[i-1][j]) /2/ dz;
        }
    }

    lapl_y_solver.solve(&psi_y[1][1], &RHS_y[1][1]);

    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            RHS_z[k][j] = (vz[k][j+1]-vz[k][j-1]) /2/ dx - (uz[k+1][j] - uz[k-1][j]) /2/ dy;
        }
    }

    lapl_z_solver.solve(&psi_z[1][1], &RHS_z[1][1]);
}


template<typename T, bool check>
void NSCube<T,check>::update_uvwp() {
#pragma omp parallel
    { // omp parallel

#pragma omp single
    { // omp single

#pragma omp task
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j < nx; j++) {
                //double r = r0+dr*j;
                u[i][k][j] = F[i][k][j]-dt/dx*(x[i][k][j+1]-x[i][k][j]);
            }
        }
    }
#pragma omp task
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k < ny; k++) {
            for (int j = 1; j <= nx; j++) {
                //double r = r0+dr*j-dr/2;
                v[i][k][j] = G[i][k][j]-dt/dy*(x[i][k+1][j]-x[i][k][j]);
            }
        }
    }
#pragma omp task
    for (int i = 1; i < nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                w[i][k][j] = H[i][k][j]-dt/dz*(x[i+1][k][j]-x[i][k][j]);
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


template<typename T, bool check>
void NSCube<T,check>::update_uvi() {
    if (time_index == plot_time_index) {
        return;
    }

    for (int k = 0; k <= ny+1; k++) {
        for (int j = 0; j <= nx+1; j++) {
            uz[k][j] = 0.5*(u[nz/2][k][j-1] + u[nz/2][k][j]);
            vz[k][j] = 0.5*(v[nz/2][k-1][j] + v[nz/2][k][j]);
        }
    }

    for (int i = 0; i <= nz+1; i++) {
        for (int j = 0; j <= nx+1; j++) {
            uy[i][j] = 0.5*(u[i][ny/2][j-1] + u[i][ny/2][j]);
            wy[i][j] = 0.5*(w[i-1][ny/2][j] + w[i][ny/2][j]);
        }
    }

    for (int i = 0; i <= nz+1; i++) {
        for (int k = 0; k <= ny+1; k++) {
            vx[i][k] = 0.5*(v[i][k-1][nx/2] + v[i][k][nx/2]);
            wx[i][k] = 0.5*(w[i-1][k][nx/2] + w[i][k][nx/2]);
        }
    }

    plot_time_index = time_index;
}

template class NSCube<double,true>;
template class NSCube<double,false>;
template class NSCube<float,true>;
template class NSCube<float,false>;

} // namespace fdm
