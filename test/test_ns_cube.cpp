#include <string>
#include <vector>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdio>

#include "tensor.h"
#include "matrix_plot.h"
#include "config.h"
#include "sparse.h"
#include "umfpack_solver.h"
#include "asp_misc.h"

using namespace std;
using namespace fdm;

using asp::format;
using asp::sq;

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

    matrix psi; // срез по плоскости Oxy
    matrix uz, vz; // срез по плоскости Oxy
    matrix uy, wy; // срез по плоскости Oxz
    matrix vx, wx; // срез по плоскости Oyz

    csr_matrix<T> P;
    bool P_initialized = false;

    umfpack_solver<T> solver;
    umfpack_solver<T> solver_stream; // для функции тока по срезу

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
        , psi({1, ny, 1, nx})

        , uz({1, ny, 1, nx}) // inner u
        , vz({1, ny, 1, nx}) // inner v

        , uy({1, nz, 1, nx})
        , wy({1, nz, 1, nx})

        , vx({1, nz, 1, ny})
        , wx({1, nz, 1, ny})
    {
        init_P();
    }

    void step() {
        // printf("1\n");
        init_bound(); // printf("2\n");
        FGH(); // printf("3\n");
        poisson(); // printf("4\n");
        update_uvwp();
        time_index++;
        //update_uvi(); // remove
        printf("%.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e \n",
               dt*time_index, p.maxabs(), u.maxabs(), v.maxabs(), w.maxabs(), x.maxabs(),
               RHS.maxabs(), F.maxabs(), G.maxabs(), H.maxabs());
    }

    void plot() {
        update_uvi();
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

        plotter.plot(matrix_plotter::page()
                     .vector(uz, vz)
                     .levels(10)
                     .labels("Y", "X", "")
                     .tlabel(format("UV (z=const) (%.1e)", dt*time_index))
                     .bounds(x1+dx/2, y1+dy/2, x2-dx/2, y2-dy/2));

        plotter.plot(matrix_plotter::page()
                     .vector(uy, wy)
                     .levels(10)
                     .labels("Z", "X", "")
                     .tlabel(format("UW (y=const) (%.1e)", dt*time_index))
                     .bounds(x1+dx/2, z1+dz/2, x2-dx/2, z2-dz/2));

        plotter.plot(matrix_plotter::page()
                     .vector(vx, wx)
                     .levels(10)
                     .labels("Z", "Y", "")
                     .tlabel(format("VW (x=const) (%.1e)", dt*time_index))
                     .bounds(y1+dy/2, z1+dz/2, y2-dy/2, z2-dz/2));
    }

    void vtk_out() {
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

private:
    void init_bound() {
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

    void FGH() {
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

    void init_P() {
        if (P_initialized) {
            return;
        }

        for (int i = 1; i <= nz; i++) {
            for (int k = 1; k <= ny; k++) {
                for (int j = 1; j <= nx; j++) {
                    int id = RHS.index({i,k,j});
                    /*if (j > 1)  { rm += rm1; }
                    if (j < nr) { rm += rm2; }
                    if (k > 1)  { zm += 1; }
                    if (k < nz) { zm += 1; }*/
                    if (i > 1) {
                        P.add(id, RHS.index({i-1,k,j}), 1/dz2);
                    }

                    if (k > 1) {
                        P.add(id, RHS.index({i,k-1,j}), 1/dy2);
                    }

                    if (j > 1) {
                        P.add(id, RHS.index({i,k,j-1}), 1/dx2);
                    }

                    P.add(id, RHS.index({i,k,j}), -2/dx2-2/dy2-2/dz2);

                    if (j < nx) {
                        P.add(id, RHS.index({i,k,j+1}), 1/dx2);
                    }

                    if (k < ny) {
                        P.add(id, RHS.index({i,k+1,j}), 1/dy2);
                    }

                    if (i < nz) {
                        P.add(id, RHS.index({i+1,k,j}), 1/dz2);
                    }
                }
            }
        }
        P.close();
        // P.sort_rows();

        solver = std::move(P);

        P_initialized = true;
    }

    void poisson() {
        for (int i = 1; i <= nz; i++) {
            for (int k = 1; k <= ny; k++) {
                for (int j = 1; j <= nx; j++) {
                    RHS[i][k][j] = ((F[i][k][j]-F[i][k][j-1])/dx
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

        solver.solve(&x[1][1][1], &RHS[1][1][1]);
    }

    void update_uvwp() {
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
        for (int i = 1; i < nz; i++) {
            for (int k = 1; k < ny; k++) {
                for (int j = 1; j < nx; j++) {
                    p[i][k][j] = x[i][k][j];
                }
            }
        }

#pragma omp taskwait
        } // end of omp single
        } // end of omp parallel
    }

    void update_uvi() {
        if (time_index == plot_time_index) {
            return;
        }

        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                uz[k][j] = 0.5*(u[nz/2][k][j-1] + u[nz/2][k][j]);
                vz[k][j] = 0.5*(v[nz/2][k-1][j] + v[nz/2][k][j]);
            }
        }

        for (int i = 1; i <= nz; i++) {
            for (int j = 1; j <= nx; j++) {
                uy[i][j] = 0.5*(u[i][ny/2][j-1] + u[i][ny/2][j]);
                wy[i][j] = 0.5*(w[i-1][ny/2][j] + w[i][ny/2][j]);
            }
        }

        for (int i = 1; i <= nz; i++) {
            for (int k = 1; k <= ny; k++) {
                vx[i][k] = 0.5*(v[i][k-1][nx/2] + v[i][k][nx/2]);
                wx[i][k] = 0.5*(w[i-1][k][nx/2] + w[i][k][nx/2]);
            }
        }

        plot_time_index = time_index;
    }
};

template<typename T, bool check>
void calc(const Config& c) {
    using namespace std::chrono;

    NSCube<T, true> ns(c);

    const int steps = c.get("ns", "steps", 1);
    const int plot_interval = c.get("plot", "interval", 100);
    const int png = c.get("plot", "png", 1);
    const int vtk = c.get("plot", "vtk", 0);
    int i;

    if (png) {
        ns.plot();
    }
    if (vtk) {
        ns.vtk_out();
    }

    auto t1 = steady_clock::now();

    for (i = 0; i < steps; i++) {
        ns.step();

        if ((i+1) % plot_interval == 0) {
            if (png) {
                ns.plot();
            }
            if (vtk) {
                ns.vtk_out();
            }
        }
    }

    auto t2 = steady_clock::now();
    auto interval = duration_cast<duration<double>>(t2 - t1);
    printf("It took me '%f' seconds\n", interval.count());
}

// Флетчер, том 2, страница 398
int main(int argc, char** argv) {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    bool check = c.get("other", "check", 0) == 1;
    if (check) {
        calc<double,true>(c);
    } else {
        calc<double,false>(c);
    }

    return 0;
}
