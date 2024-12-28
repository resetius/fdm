#include "velocity_plot.h"
#include "matrix_plot.h"
#include "asp_misc.h"

using namespace asp;
using std::vector;
using std::string;

namespace fdm {

template<typename T, bool check, typename F>
void velocity_plotter<T,check,F>::use(T* u0, T* v0, T* w0)
{
    u.use(u0); v.use(v0); w.use(w0);
}

template<typename T, bool check, typename F>
void velocity_plotter<T,check,F>::update()
{
    for (int i = z0; i <= znn; i++) {
        for (int k = y0; k <= ynn; k++) {
            vx[i][k] = 0.5*(v[i][k-1][nx/2] + v[i][k][nx/2]);
            wx[i][k] = 0.5*(w[i-1][k][nx/2] + w[i][k][nx/2]);
        }
    }

    for (int i = z0; i <= znn; i++) {
        for (int j = 0; j <= nx+1; j++) {
            uy[i][j] = 0.5*(u[i][ny/2][j-1] + u[i][ny/2][j]);
            wy[i][j] = 0.5*(w[i-1][ny/2][j] + w[i][ny/2][j]);
        }
    }

    for (int k = y0; k <= ynn; k++) {
        for (int j = 0; j <= nx+1; j++) {
            uz[k][j] = 0.5*(u[nz/2][k][j-1] + u[nz/2][k][j]);
            vz[k][j] = 0.5*(v[nz/2][k-1][j] + v[nz/2][k][j]);
        }
    }

    for (int i = z1; i <= zn; i++) {
        for (int k = y1; k <= yn; k++) {
            RHS_x[i][k] = (wx[i][k+1]-wx[i][k-1]) /2/ dy - (vx[i+1][k] - vx[i-1][k]) /2/ dz;
        }
    }

    lapl_x.solve(&psi_x[z1][y1], &RHS_x[z1][y1]);

    for (int i = z1; i <= zn; i++) {
        for (int j = 1; j <= nx; j++) {
            RHS_y[i][j] = (wy[i][j+1]-wy[i][j-1]) /2/ dx - (uy[i+1][j] - uy[i-1][j]) /2/ dz;
        }
    }

    lapl_y.solve(&psi_y[z1][1], &RHS_y[z1][1]);

    for (int k = y1; k <= yn; k++) {
        for (int j = 1; j <= nx; j++) {
            RHS_z[k][j] = (vz[k][j+1]-vz[k][j-1]) /2/ dx - (uz[k+1][j] - uz[k-1][j]) /2/ dy;
        }
    }

    lapl_z.solve(&psi_z[y1][1], &RHS_z[y1][1]);
}

template<typename T, bool check, typename F>
void velocity_plotter<T,check,F>::plot(const string& name, double t)
{
    matrix_plotter plotter(matrix_plotter::settings()
                           .sub(2, 3)
                           .devname("pngcairo")
                           .fname(name));

    plotter.plot(matrix_plotter::page()
                 .scalar(uz)
                 .levels(10)
                 .labels(Ylabel, Xlabel, "")
                 .tlabel(format("U (t=%.1e, |max|=%.1e)", t, uz.maxabs()))
                 .bounds(xx1, yy1, xx2, yy2));
    plotter.plot(matrix_plotter::page()
                 .scalar(vz)
                 .levels(10)
                 .labels(Ylabel, Xlabel, "")
                 .tlabel(format("V (t=%.1e, |max|=%.1e)", t, vz.maxabs()))
                 .bounds(xx1, yy1, xx2, yy2));
    plotter.plot(matrix_plotter::page()
                 .scalar(wy)
                 .levels(10)
                 .labels(Zlabel, Xlabel, "")
                 .tlabel(format("W (t=%.1e, |max|=%.1e)", t, wy.maxabs()))
                 .bounds(xx1, zz1, xx2, zz2));
    plotter.plot(matrix_plotter::page()
                 .scalar(psi_z)
                 .levels(10)
                 .labels(Ylabel, Xlabel, "")
                 .tlabel(format("UV (%s=const) (%.1e)", Zlabel.c_str(), t))
                 .bounds(xx1, yy1, xx2, yy2));
    plotter.plot(matrix_plotter::page()
                 .scalar(psi_y)
                 .levels(10)
                 .labels(Zlabel, Xlabel, "")
                 .tlabel(format("UW (%s=const) (%.1e)", Ylabel.c_str(), t))
                 .bounds(xx1, zz1, xx2, zz2));
    plotter.plot(matrix_plotter::page()
                 .scalar(psi_x)
                 .levels(10)
                 .labels(Zlabel, Ylabel, "")
                 .tlabel(format("VW (%s=const) (%.1e)", Xlabel.c_str(), t))
                 .bounds(yy1, zz1, yy2, zz2));
}

template<typename T, bool check, typename F>
void velocity_plotter<T,check,F>::vtk_out(const string& name, int time_index)
{
    FILE* f = fopen(name.c_str(), "wb");
    // not cyl only yet
    fprintf(f, "# vtk DataFile Version 3.0\n");
    fprintf(f, "step %d\n", time_index);
    fprintf(f, "ASCII\n");

    if (cyl) {
        verify(zflag == tensor_flag::periodic);

        fprintf(f, "DATASET UNSTRUCTURED_GRID\n");
        fprintf(f, "POINTS %d double\n", (nx+1)*(yn+1)*nz);
        for (int i = 0; i < nz; i++) {
            for (int k = 0; k < ny+1; k++) {
                for (int j = 0; j < nx+1; j++) {
                    double r = xx1+dx*j;
                    double z = yy1+dy*k;
                    double phi = dz*i;

                    double x = r*cos(phi);
                    double y = r*sin(phi);

                    fprintf(f, "%f %f %f\n", x, y, z);
                }
            }
        }
        int l = nz*ny*nx;
        fprintf(f, "CELLS %d %d\n", l, 9*l);
        for (int i = 0; i < nz; i++) {
            for (int k = 0; k < ny; k++) {
                for (int j = 0; j < nx; j++) {
                    //{i,k,j}     - {i+1,k,j}
                    //{i+1,k,j}   - {i+1,k,j+1}
                    //{i+1,k,j+1} - {i,k,j+1}

                    // up   {i,k,j}   - {i+1,k,j}   - {i+1,k,j+1}   - {i,k,j+1}
                    // down {i,k+1,j} - {i+1,k+1,j} - {i+1,k+1,j+1} - {i,k+1,j+1}

#define Id(i,k,j) ((i)%nz)*(nx+1)*(ny+1)+(k)*(nx+1)+(j)
                    fprintf(f, "8 %d %d %d %d %d %d %d %d\n",
                            Id(i,k,j),   Id(i+1,k,j),   Id(i+1,k,j+1),   Id(i,k,j+1),
                            Id(i,k+1,j), Id(i+1,k+1,j), Id(i+1,k+1,j+1), Id(i,k+1,j+1)
                        );
#undef Id
                }
            }
        }
        fprintf(f, "CELL_TYPES %d\n", l);
        for (int i = 0; i < nz; i++) {
            for (int k = 0; k < ny; k++) {
                for (int j = 0; j < nx; j++) {
                    // VTK_HEXAHEDRON
                    fprintf(f, "12\n");
                }
            }
        }
        fprintf(f, "CELL_DATA %d\n", nx*ny*nz);
        fprintf(f, "VECTORS u double\n");

        for (int i = 0; i < nz; i++) {
            for (int k = y1; k <= yn; k++) {
                for (int j = 1; j <= nx; j++) {
                    double phi = dz*i+dz/2;
                    double r = xx1+dx*j+dx/2;

                    double u0 = 0.5*(u[i][k][j]+u[i][k][j-1]);
                    double v0 = 0.5*(v[i][k][j]+v[i][k-1][j]);
                    double w0 = 0.5*(w[i][k][j]+w[i-1][k][j]);

                    double l = sqrt(u0*u0+v0*v0+r*r*w0*w0);
                    u0 /= l; v0 /= l; w0 /= l;

                    double x = u0 * cos(phi) - w0 * sin(phi);
                    double y = u0 * sin(phi) + w0 * cos(phi);
                    double z = v0;
                    if (std::abs(l) < 1e-7) {
                        l = 1e-4; // hack
                    }
                    x *= l; y *=l; z *= l;

                    fprintf(f, "%f %f %f\n", x, y, z);
                }
            }
        }

    } else {
        fprintf(f, "DATASET STRUCTURED_POINTS\n");
        fprintf(f, "DIMENSIONS %d %d %d\n", nx, ny, nz);
        fprintf(f, "ASPECT_RATIO 1 1 1\n");
        fprintf(f, "ORIGIN %f %f %f\n", xx1, yy1, zz1); // TODO: check
        fprintf(f, "SPACING %f %f %f\n", dx, dy, dz); // TODO: check
        fprintf(f, "POINT_DATA %d\n", nx*ny*nz);
        fprintf(f, "VECTORS u double\n");
        for (int i = z1; i <= zn; i++) {
            for (int k = y1; k <= yn; k++) {
                for (int j = 1; j <= nx; j++) {
                    fprintf(f, "%f %f %f\n",
                            0.5*(u[i][k][j]+u[i][k][j-1]),
                            0.5*(v[i][k][j]+v[i][k-1][j]),
                            0.5*(w[i][k][j]+w[i-1][k][j])
                        );
                }
            }
        }
    }
    fclose(f);
}

template class velocity_plotter<double,true,tensor_flags<>>;
template class velocity_plotter<double,false,tensor_flags<>>;
template class velocity_plotter<float,true,tensor_flags<>>;
template class velocity_plotter<float,false,tensor_flags<>>;

template class velocity_plotter<double,true,tensor_flags<tensor_flag::periodic>>;
template class velocity_plotter<double,false,tensor_flags<tensor_flag::periodic>>;
template class velocity_plotter<float,true,tensor_flags<tensor_flag::periodic>>;
template class velocity_plotter<float,false,tensor_flags<tensor_flag::periodic>>;

template class velocity_plotter<double,true,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;
template class velocity_plotter<double,false,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;
template class velocity_plotter<float,true,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;
template class velocity_plotter<float,false,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;

} // namespace fdm
