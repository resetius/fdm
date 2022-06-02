#include <string>
#include <vector>
#include <climits>
#include <cmath>

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
class NSCyl {
public:
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;
    using tensor = fdm::tensor<T,3,check,tensor_flags>;
    using matrix = fdm::tensor<T,2,check>;
    using matrix_p = fdm::tensor<T,2,check,tensor_flags>;

    const double R, r0;
    const double h1, h2;
    const double U0; // скорость вращения внутреннего цилиндра

    const double Re;
    const double dt;

    const int nr, nz, nphi;
    const double dr, dz, dphi;
    const double dr2, dz2, dphi2;

    tensor u /*r*/,v/*z*/,w/*phi*/;
    tensor p,x;
    tensor F,G,H,RHS;

    matrix_p RHS_r,RHS_z;
    matrix RHS_phi;

    matrix_p psi_r, psi_z;
    matrix psi_phi; // срезы

    matrix uphi, vphi;
    matrix_p uz, wz;
    matrix_p vr, wr;

    csr_matrix<T> P;
    csr_matrix<T> P_r, P_z, P_phi;
    bool P_initialized = false;

    umfpack_solver<T> solver;
    umfpack_solver<T> solver_stream_r; // для функции тока по срезу
    umfpack_solver<T> solver_stream_z; // для функции тока по срезу
    umfpack_solver<T> solver_stream_phi; // для функции тока по срезу

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
        , nz(c.get("ns", "nz", 32))
        , nphi(c.get("ns", "nphi", 32))
        , dr((R-r0)/nr), dz((h2-h1)/nz), dphi(2*M_PI/nphi)
        , dr2(dr*dr), dz2(dz*dz), dphi2(dphi*dphi)

          // phi, z, r
        , u{{0, nphi-1, 0, nz+1, -1, nr+1}} // check bounds
        , v{{0, nphi-1, -1, nz+1, 0, nr+1}} // check bounds
        , w{{0, nphi-1, 0, nz+1, 0, nr+1}} // check bounds
        , p({0, nphi-1, 0, nz+1, 0, nr+1})

        , x({0, nphi-1, 1, nz, 1, nr})
        , F({0, nphi-1, 1, nz, 0, nr}) // check bounds
        , G({0, nphi-1, 0, nz, 1, nr}) // check bounds
        , H({0, nphi-1, 1, nz, 1, nr}) // check bounds
        , RHS({0, nphi-1, 1, nz, 1, nr})

        , RHS_r({0, nphi-1, 1, nz})
        , RHS_z({0, nphi-1, 1, nr})
        , RHS_phi({1, nz, 1, nr})

        , psi_r({0, nphi-1, 1, nz})
        , psi_z({0, nphi-1, 1, nr})
        , psi_phi({1, nz, 1, nr})

        , uphi({0, nz, 0, nr})
        , vphi({0, nz, 0, nr})

        , uz({0, nphi-1, 0, nr})
        , wz({0, nphi-1, 0, nr})

        , vr({0, nphi-1, 0, nz})
        , wr({0, nphi-1, 0, nz})
    {
        init_P();
        init_P_slices();
    }

    void step() {
        init_bound();
        FGH();
        poisson();
        update_uvwp();
        time_index++;
        //update_uvi(); // remove
        printf("%.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e %.1e \n",
               p.maxabs(), u.maxabs(), v.maxabs(), w.maxabs(), x.maxabs(),
               RHS.maxabs(), F.maxabs(), G.maxabs(), H.maxabs());
    }

    void plot() {
        update_uvi();
        poisson_stream();
        matrix_plotter plotter(matrix_plotter::settings()
                               .sub(2, 3)
                               .devname("pngcairo")
                               .fname(format("step_%07d.png", time_index)));

        plotter.plot(matrix_plotter::page()
                     .scalar(uphi)
                     .levels(10)
                     .labels("Z", "R", "")
                     .tlabel(format("U (t=%.1e, |max|=%.1e)", dt*time_index, uphi.maxabs()))
                     .bounds(r0+dr/2, h1+dz/2, R-dr/2, h2-dz/2));
        plotter.plot(matrix_plotter::page()
                     .scalar(vphi)
                     .levels(10)
                     .labels("Z", "R", "")
                     .tlabel(format("V (t=%.1e, |max|=%.1e)", dt*time_index, vphi.maxabs()))
                     .bounds(r0+dr/2, h1+dz/2, R-dr/2, h2-dz/2));
        plotter.plot(matrix_plotter::page()
                     .scalar(wz)
                     .levels(10)
                     .labels("PHI", "R", "")
                     .tlabel(format("W (t=%.1e, |max|=%.1e)", dt*time_index, wz.maxabs()))
                     .bounds(r0+dr/2, 0, R-dr/2, 2*M_PI));

        plotter.plot(matrix_plotter::page()
                     .scalar(psi_phi)
                     .levels(10)
                     .labels("Z", "R", "")
                     .tlabel(format("UV (phi=const) (%.1e)", dt*time_index))
                     .bounds(r0+dr/2, h1+dz/2, R-dr/2, h2-dz/2));
        plotter.plot(matrix_plotter::page()
                     .scalar(psi_z)
                     .levels(10)
                     .labels("Phi", "R", "")
                     .tlabel(format("UW (z=const) (%.1e)", dt*time_index))
                     .bounds(r0+dr/2, 0, R-dr/2, 2*M_PI));
        plotter.plot(matrix_plotter::page()
                     .scalar(psi_r)
                     .levels(10)
                     .labels("Phi", "Z", "")
                     .tlabel(format("VW (r=const) (%.1e)", dt*time_index))
                     .bounds(h1+dz/2, 0, h2-dz/2, 2*M_PI));

    }

    void vtk_out() {
        update_uvi();

        FILE* f = fopen(format("step_%07d.vtk", time_index).c_str(), "wb");
        fprintf(f, "# vtk DataFile Version 3.0\n");
        fprintf(f, "step %d\n", time_index);
        fprintf(f, "ASCII\n");
        fprintf(f, "DATASET UNSTRUCTURED_GRID\n");
        //fprintf(f, "DIMENSIONS %d %d %d\n", nr, nz, nphi);
        fprintf(f, "POINTS %d double\n", (nr+1)*(nz+1)*nphi);
        for (int i = 0; i < nphi; i++) {
            for (int k = 0; k < nz+1; k++) {
                for (int j = 0; j < nr+1; j++) {
                    double r = r0+dr*j;
                    double z = h1+dz*k;
                    double phi = dphi*i;

                    double x = r*cos(phi);
                    double y = r*sin(phi);

                    fprintf(f, "%f %f %f\n", x, y, z);
                }
            }
        }
        int l = nphi*nz*nr;
        fprintf(f, "CELLS %d %d\n", l, 9*l);
        for (int i = 0; i < nphi; i++) {
            for (int k = 0; k < nz; k++) {
                for (int j = 0; j < nr; j++) {
                    //{i,k,j}     - {i+1,k,j}
                    //{i+1,k,j}   - {i+1,k,j+1}
                    //{i+1,k,j+1} - {i,k,j+1}

                    // up   {i,k,j}   - {i+1,k,j}   - {i+1,k,j+1}   - {i,k,j+1}
                    // down {i,k+1,j} - {i+1,k+1,j} - {i+1,k+1,j+1} - {i,k+1,j+1}

#define Id(i,k,j) ((i)%nphi)*(nr+1)*(nz+1)+(k)*(nr+1)+(j)
                    fprintf(f, "8 %d %d %d %d %d %d %d %d\n",
                            Id(i,k,j),   Id(i+1,k,j),   Id(i+1,k,j+1),   Id(i,k,j+1),
                            Id(i,k+1,j), Id(i+1,k+1,j), Id(i+1,k+1,j+1), Id(i,k+1,j+1)
                        );
#undef Id
                }
            }
        }

        fprintf(f, "CELL_TYPES %d\n", l);
        for (int i = 0; i < nphi; i++) {
            for (int k = 0; k < nz; k++) {
                for (int j = 0; j < nr; j++) {
                    // VTK_HEXAHEDRON
                    fprintf(f, "12\n");
                }
            }
        }

        fprintf(f, "CELL_DATA %d\n", nr*nz*nphi);
        fprintf(f, "VECTORS u double\n");

        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    double phi = dphi*i+dphi/2;
                    double r = r0+dr*j+dr/2;

                    double u0 = 0.5*(u[i][k][j]+u[i][k][j-1]);
                    double v0 = 0.5*(v[i][k][j]+v[i][k-1][j]);
                    double w0 = 0.5*(w[i][k][j]+w[i-1][k][j]);

                    double l = sqrt(u0*u0+v0*v0+r*r*w0*w0);
                    u0 /= l; v0 /= l; w0 /= l;

                    double x = u0 * cos(phi) - w0 * sin(phi);
                    double y = u0 * sin(phi) + w0 * cos(phi);
                    double z = v0;
                    x *= l; y *=l; z *= l;

                    fprintf(f, "%f %f %f\n", x, y, z);
                }
            }
        }

        fclose(f);
    }

private:
    void init_bound() {
        // внутренний цилиндр
        for (int i = 0; i < nphi; i++) {
            for (int k = 0; k <= nz+1; k++) {
                // 0.5*(w[i][k][0] + w[i][k][1]) = U0
                w[i][k][0] = 2*U0 - w[i][k][1]; // inner
                w[i][k][nr+1] = -w[i][k][nr]; // outer
            }
        }
        for (int i = 0; i < nphi; i++) {
            for (int k = -1; k <= nz+1; k++) {
                v[i][k][0]    = - v[i][k][1]; // inner
                v[i][k][nr+1] = -v[i][k][nr]; // outer
            }
        }

        // div=0 на границе
        // инициализация узлов за пределами области
        for (int i = 0; i < nphi; i++) {
            for (int k = 0; k <= nz+1; k++) {
                u[i][k][-1]   = u[i][k][1];
                u[i][k][nr+1] = u[i][k][nr-1];
            }
        }

        for (int i = 0; i < nphi; i++) {
            for (int j = 0; j <= nr+1; j++) {
                v[i][-1][j]   = v[i][1][j];
                v[i][nz+1][j] = v[i][nz-1][j];
            }
        }

        // давление
        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                p[i][k][0]  = p[i][k][1] -
                    (u[i][k][1]-2*u[i][k][0]+u[i][k][-1])/Re/dr;
                p[i][k][nr+1] = p[i][k][nr]-
                    (u[i][k][nr+1]-2*u[i][k][nr]+u[i][k][nr-1])/Re/dr;
            }
        }
        for (int i = 0; i < nphi; i++) {
            for (int j = 1; j <= nr; j++) {
                p[i][0][j]    = p[i][1][j] -
                    (v[i][1][j]-2*v[i][0][j]+v[i][-1][j])/Re/dz;
                p[i][nz+1][j] = p[i][nz][j]-
                    (v[i][nz+1][j]-2*v[i][nz][j]+v[i][nz-1][j])/Re/dz;
            }
        }
    }

    void FGH() {
        // F (r)
        for (int i = 1; i <= nphi; i++) {
            for (int k = 1; k <= nz; k++) { // 3/2 ..
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
        for (int i = 1; i <= nphi; i++) {
            for (int k = 0; k <= nz; k++) {
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
        for (int i = 0; i < nphi; i++) { // 1/2 ...
            for (int k = 1; k <= nz; k++) {
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
    }

    void init_P() {
        if (P_initialized) {
            return;
        }

        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    int id = RHS.index({i,k,j});
                    double r = r0+j*dr-dr/2;
                    double r2 = r*r;
                    //double rm1 = (r-0.5*dr)/r;
                    //double rm2 = (r+0.5*dr)/r;
                    double rm = 0;
                    double zm = 0;

                    /*if (j > 1)  { rm += rm1; }
                    if (j < nr) { rm += rm2; }
                    if (k > 1)  { zm += 1; }
                    if (k < nz) { zm += 1; }*/
                    zm = 2; rm = 2;

                    P.add(id, RHS.index({i-1,k,j}), 1/dphi2/r2);

                    if (k > 1) {
                        P.add(id, RHS.index({i,k-1,j}), 1/dz2);
                    }

                    if (j > 1) {
                        P.add(id, RHS.index({i,k,j-1}), (r-0.5*dr)/dr2/r);
                    }

                    P.add(id, RHS.index({i,k,j}), -rm/dr2-zm/dz2-2/dphi2/r2);

                    if (j < nr) {
                        P.add(id, RHS.index({i,k,j+1}), (r+0.5*dr)/dr2/r);
                    }

                    if (k < nz) {
                        P.add(id, RHS.index({i,k+1,j}), 1/dz2);
                    }

                    P.add(id, RHS.index({i+1,k,j}), 1/dphi2/r2);
                }
            }
        }
        P.close();
        P.sort_rows();

        solver = std::move(P);

        P_initialized = true;
    }

    void init_P_slices() {
        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                int id = RHS_r.index({i,k});
                double r = r0-dr/2;
                double r2 = r*r;

                P_r.add(id, RHS_r.index({i-1,k}), 1/dphi2/r2);

                if (k > 1) {
                    P_r.add(id, RHS_r.index({i,k-1}), 1/dz2);
                }

                P_r.add(id, RHS_r.index({i,k}), -2/dz2-2/dphi2/r2);

                if (k < nz) {
                    P_r.add(id, RHS_r.index({i,k+1}), 1/dz2);
                }

                P_r.add(id, RHS_r.index({i+1,k}), 1/dphi2/r2);
            }
        }
        P_r.close();
        P_r.sort_rows();

        solver_stream_r = std::move(P_r);

        for (int i = 0; i < nphi; i++) {
            for (int j = 1; j <= nr; j++) {
                int id = RHS_z.index({i,j});
                double r = r0+j*dr-dr/2;
                double r2 = r*r;
                double rm = 2;

                P_z.add(id, RHS_z.index({i-1,j}), 1/dphi2/r2);

                if (j > 1) {
                    P_z.add(id, RHS_z.index({i,j-1}), (r-0.5*dr)/dr2/r);
                }

                P_z.add(id, RHS_z.index({i,j}), -rm/dr2-2/dphi2/r2);

                if (j < nr) {
                    P_z.add(id, RHS_z.index({i,j+1}), (r+0.5*dr)/dr2/r);
                }

                P_z.add(id, RHS_z.index({i+1,j}), 1/dphi2/r2);
            }
        }
        P_z.close();
        P_z.sort_rows();

        solver_stream_z = std::move(P_z);

        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                int id = RHS_phi.index({k,j});
                double r = r0+j*dr-dr/2;
                double rm = 2;
                double zm = 2;

                if (k > 1) {
                    P_phi.add(id, RHS_phi.index({k-1,j}), 1/dz2);
                }

                if (j > 1) {
                    P_phi.add(id, RHS_phi.index({k,j-1}), (r-0.5*dr)/dr2/r);
                }

                P_phi.add(id, RHS_phi.index({k,j}), -rm/dr2-zm/dz2);

                if (j < nr) {
                    P_phi.add(id, RHS_phi.index({k,j+1}), (r+0.5*dr)/dr2/r);
                }

                if (k < nz) {
                    P_phi.add(id, RHS_phi.index({k+1,j}), 1/dz2);
                }
            }
        }
        P_phi.close();
        P_phi.sort_rows();

        solver_stream_phi = std::move(P_phi);
    }

    void poisson() {
        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    double r = r0+dr*j-dr/2;

                    RHS[i][k][j] = (((r+0.5*dr)*F[i][k][j]-(r-0.5*dr)*F[i][k][j-1])/r/dr
                                    +(G[i][k][j]-G[i][k-1][j])/dz
                                    +(H[i][k][j]-H[i-1][k][j])/dphi/r)/dt;

                    if (k <= 1) {
                        RHS[i][k][j] -= p[i][k-1][j]/dz2;
                    }
                    if (j <= 1) {
                        RHS[i][k][j] -= (r-dr/2)/r*p[i][k][j-1]/dr2;
                    }


                    if (j >= nr) {
                        RHS[i][k][j] -= (r+dr/2)/r*p[i][k][j+1]/dr2;
                    }
                    if (k >= nz) {
                        RHS[i][k][j] -= p[i][k+1][j]/dz2;
                    }
                }
            }
        }

        solver.solve(&x[0][1][1], &RHS[0][1][1]);
    }

    void poisson_stream() {
        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                RHS_r[i][k] = (wr[i][k]-wr[i][k-1]) / dz - (vr[i][k] - vr[i-1][k]) / dphi;
            }
        }

        solver_stream_r.solve(&psi_r[0][1], &RHS_r[0][1]);

        for (int i = 0; i < nphi; i++) {
            for (int j = 1; j <= nr; j++) {
                RHS_z[i][j] = (wz[i][j]-wz[i][j-1]) / dr - (uz[i][j] - uz[i-1][j]) / dphi;
            }
        }

        solver_stream_z.solve(&psi_z[0][1], &RHS_z[0][1]);

        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                RHS_phi[k][j] = (vphi[k][j]-vphi[k][j-1]) / dr - (uphi[k][j] - uphi[k-1][j]) / dz;
            }
        }

        solver_stream_phi.solve(&psi_phi[1][1], &RHS_phi[1][1]);
    }

    void update_uvwp() {
        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j < nr; j++) {
                    //double r = r0+dr*j;
                    u[i][k][j] = F[i][k][j]-dt/dr*(x[i][k][j+1]-x[i][k][j]);
                }
            }
        }

        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k < nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    //double r = r0+dr*j-dr/2;
                    v[i][k][j] = G[i][k][j]-dt/dz*(x[i][k+1][j]-x[i][k][j]);
                }
            }
        }

        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    double r = r0+dr*j-dr/2;
                    w[i][k][j] = H[i][k][j]-dt/dphi/r*(x[i+1][k][j]-x[i][k][j]);
                }
            }
        }

        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k < nz; k++) {
                for (int j = 1; j < nr; j++) {
                    p[i][k][j] = x[i][k][j];
                }
            }
        }
    }

    void update_uvi() {
        if (time_index == plot_time_index) {
            return;
        }

        for (int k = 0; k <= nz; k++) {
            for (int j = 0; j <= nr; j++) {
                uphi[k][j] = 0.5*(u[nphi/2][k][j-1] + u[nphi/2][k][j]);
                vphi[k][j] = 0.5*(v[nphi/2][k-1][j] + v[nphi/2][k][j]);
            }
        }

        for (int i = 0; i < nphi; i++) {
            for (int j = 0; j <= nr; j++) {
                uz[i][j] = 0.5*(u[i][nz/2][j-1] + u[i][nz/2][j]);
                wz[i][j] = 0.5*(w[i-1][nz/2][j] + w[i][nz/2][j]);
            }
        }

        for (int i = 0; i < nphi; i++) {
            for (int k = 0; k <= nz; k++) {
                vr[i][k] = 0.5*(v[i][k-1][nr/2] + v[i][k][nr/2]);
                wr[i][k] = 0.5*(w[i-1][k][nr/2] + w[i][k][nr/2]);
            }
        }

        plot_time_index = time_index;
    }
};

template<typename T, bool check>
void calc(const Config& c) {
    NSCyl<T, true> ns(c);

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
