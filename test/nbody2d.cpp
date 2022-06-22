#include <random>

#include "config.h"
#include "tensor.h"
#include "blas.h"
#include "lapl_rect.h"
#include "matrix_plot.h"
#include "asp_misc.h"

using namespace fdm;
using namespace std;
using namespace asp;

template<typename T,bool check,tensor_flag flag>
class NBody {
public:
    using flags = typename short_flags<flag,flag>::value;
    using tensor =  fdm::tensor<T,2,check,flags>;

    double origin[2];
    double l;
    int n, N;
    double h;
    double dt;
    double G;
    double vel;
    int sgn;
    double rcrit;

    struct Body {
        T x[2];
        T v[2];
        T mass;

        int k;
        int j;

        T F[2]; // Local force
    };

    struct Cell {
        vector<int> bodies;
        vector<int> next;
    };

    vector<Body> bodies;
    int n0,n1,nn,nnn;
    tensor psi, rhs, f;
    fdm::tensor<Cell,2,check,flags> cells;

    LaplRectFFT2<T,check,flags> solver;

    NBody(double x0, double y0, double l, int n, int N, double dt, double G, double vel, int sgn)
        : origin{x0, y0}
        , l(l)
        , n(n) // n+1 - число ячеек (для сдвинутых сеток n - число ячеек)
        , N(N) // число тел
        , h(l / (n+1))
        , dt(dt)
        , G(G)
        , vel(vel)
        , sgn(sgn)
        , rcrit(h/4)
        , bodies(N)
        , n0(0)
        , n1(flag==tensor_flag::periodic?0:1)
        , nn(flag==tensor_flag::periodic?n-1:n)
        , nnn(flag==tensor_flag::periodic?n-1:n+1)
        , psi({n1,nn,n1,nn})
        , rhs({n1,nn,n1,nn})
        , f({n0,nnn,n0,nnn})
        , cells({n0,nn,n0,nn})
        , solver(h,h,l,l,n,n)
    {
        init_points();
    }

    void step() {
        // 1. mass to edges
#pragma omp parallel for
        for (int k = n0; k <= nnn; k++) {
            for (int j = n0; j <= nnn; j++) {
                f[k][j] = 0;
            }
        }

#pragma omp parallel for
        for (auto& body : bodies) {
            int k,j;

            double m = body.mass;
            double x = body.x[0]-origin[0];
            double y = body.x[1]-origin[1];

            if (x < 0 || x > l) continue;
            if (y < 0 || y > l) continue;

            j = floor(x / h);
            k = floor(y / h);

            x = (x-j*h)/h;
            y = (y-k*h)/h;

            verify(0 <= x && x <= 1);
            verify(0 <= y && y <= 1);

            f[k][j]       += m * (1-y)*(1-x);
            f[k][j+1]     += m * (1-y)*(x);
            f[k+1][j]     += m * (y)*(1-x);
            f[k+1][j+1]   += m * (y)*(x);
        }

        // -4pi G ro
#pragma omp parallel for
        for (int k = n1; k <= nn; k++) {
            for (int j = n1; j <= nn; j++) {
                rhs[k][j] = sgn* 4.*G*M_PI*f[k][j];
            }
        }
        solver.solve(&psi[n1][n1], &rhs[n1][n1]);

#pragma omp parallel for
        for (int k = n0; k <= nn; k++) {
            for (int j = n0; j <= nn; j++) {
                cells[k][j].next.clear();
            }
        }

#pragma omp parallel for
        for (auto& body : bodies) {
            for (int i = 0; i < 2; i++) {
                body.F[i] = 0;
            }
        }

#pragma omp parallel for
        for (int k = n0; k <= nn; k++) {
            for (int j = n0; j <= nn; j++) {
                auto& cell = cells[k][j];

                // local forces
                calc_local_forces(cell);

                for (int k0 = -1; k0 <= 1; k0++) {
                    for (int j0 = -1; j0 <= 1; j0++) {
                        if (k0 == 0 && j0 == 0) continue;

                        if constexpr(flag == tensor_flag::periodic) {
                            calc_local_forces(cell, cells[k+k0][j+j0]);
                        } else {
                            if (k+k0 >= 0 && k+k0 <= nn && j+j0 >=0 && j+j0 <= nn) {
                                calc_local_forces(cell, cells[k+k0][j+j0]);
                            }
                        }
                    }
                }
            }
        }

#pragma omp parallel for
        for (int k = n0; k <= nn; k += 2) {
            for (int j = n0; j <= nn; j += 2) {
                calc_cell(cells[k][j]);
            }
        }

#pragma omp parallel for
        for (int k = n0+1; k <= nn; k += 2) {
            for (int j = n0; j <= nn; j += 2) {
                calc_cell(cells[k][j]);
            }
        }

#pragma omp parallel for
        for (int k = n0; k <= nn; k += 2) {
            for (int j = n0+1; j <= nn; j += 2) {
                calc_cell(cells[k][j]);
            }
        }

#pragma omp parallel for
        for (int k = n0+1; k <= nn; k += 2) {
            for (int j = n0+1; j <= nn; j += 2) {
                calc_cell(cells[k][j]);
            }
        }

#pragma omp parallel for
        for (int k = n0; k <= nn; k++) {
            for (int j = n0; j <= nn; j++) {
                cells[k][j].next.swap(cells[k][j].bodies);
            }
        }

        int k = 10;
        printf("%e %e \n", bodies[k].x[0],bodies[k].x[1]);
    }

    void plot(int step) {
        string fname = format("step_%07d.png", step);
        matrix_plotter plotter(matrix_plotter::settings()
                               .sub(1, 2)
                               .devname("pngcairo")
                               .fname(fname));

        plotter.plot(matrix_plotter::page()
                     .scalar(f)
                     .levels(10)
                     .tlabel("F")
                     .bounds(origin[0], origin[1], origin[0]+l, origin[1]+l));

        plotter.plot(matrix_plotter::page()
                     .scalar(psi)
                     .levels(10)
                     .tlabel("Psi")                     .bounds(origin[0], origin[1], origin[0]+l, origin[1]+l));
    }

private:
    void apply_bi_bj(Body& bi, Body& bj, bool apply_bj) {
        double eps = 0.01;
        double R = 0;
        for (int k = 0; k < 2; k++) {
            R += sq(bi.x[k]-bj.x[k]);
        }
        R = std::sqrt(R)+eps;

        if (R < rcrit) {
            for (int k = 0; k < 2; k++) {
                bi.F[k] += sgn * bj.mass * G * (bi.x[k] - bj.x[k]) /R/R/R;
            }

            if (apply_bj) {
                for (int k = 0; k < 2; k++) {
                    bj.F[k] += sgn * bi.mass * G * (bj.x[k] - bi.x[k]) /R/R/R;
                }
            }
        }
    }

    void calc_local_forces(Cell& cell) {
        int nbodies = static_cast<int>(cell.bodies.size());

        for (int i = 0; i < nbodies; i++) {
            for (int j = i+1; j < nbodies; j++) {
                auto& bi = bodies[cell.bodies[i]];
                auto& bj = bodies[cell.bodies[j]];

                apply_bi_bj(bi, bj, true);
            }
        }
    }

    void calc_local_forces(Cell& cell, Cell& other) {
        for (auto& bi : cell.bodies) {
            for (auto& bj : other.bodies) {
                apply_bi_bj(bodies[bi], bodies[bj], false);
            }
        }
    }

    void calc_cell(Cell& cell) {
        for (auto& index : cell.bodies) {
            auto& body = bodies[index];
            if constexpr(flag == tensor_flag::none) {
                if (body.k == 0 || body.j == 0 || body.k == n+1 || body.j == n+1) {
                    continue;
                }
            }

            int k = body.k; int j = body.j;
            double a[2];

            // TODO: better interpolate force
            a[0] = (psi[k][j+1]-psi[k][j-1])/2/h + body.F[0];
            a[1] = (psi[k+1][j]-psi[k-1][j])/2/h + body.F[1];

            for (int m = 0; m < 2; m++) {
                body.x[m] += dt * body.v[m];
                body.v[m] += dt * a[m];

                if constexpr(flag == tensor_flag::periodic) {
                    if (body.x[m] < origin[m]) {
                        body.x[m] += l;
                    }
                    if (body.x[m] >= origin[m]+l) {
                        body.x[m] -= l;
                    }
                }
            }

            double x = body.x[0]-origin[0];
            double y = body.x[1]-origin[1];

            if (x < 0 || x > l) continue;
            if (y < 0 || y > l) continue;

            int next_j = floor(x / h);
            int next_k = floor(y / h);

            verify(abs(body.k-next_k) <= 1, "Too fast, try decrease dt");
            verify(abs(body.j-next_j) <= 1, "Too fast, try decrease dt");
            body.k = next_k; body.j = next_j;

            cells[body.k][body.j].next.push_back(index);
        }
    }

    void init_points() {
        std::default_random_engine generator;
        std::uniform_real_distribution<T> distribution(0, l/2);

        T minx = 10000;
        T maxx = -10000;
        T miny = 10000;
        T maxy = -10000;

        for (int index = 0; index < static_cast<int>(bodies.size()); index++) {
            auto& body = bodies[index];
            for (int i = 0; i < 2; i++) {
                body.x[i] = distribution(generator) + origin[i] + l/4;
            }
            body.mass = 0.2 + 1.5 * distribution(generator) / l;

            body.j = floor((body.x[0]-origin[0]) / h);
            body.k = floor((body.x[1]-origin[1]) / h);
            cells[body.k][body.j].bodies.push_back(index);

            minx = min(minx, body.x[0]);
            maxx = max(maxx, body.x[0]);
            miny = min(miny, body.x[1]);
            maxy = max(maxy, body.x[1]);

            double R = std::sqrt(blas::dot(3, body.x, 1, body.x, 1));
            double V = vel/sqrt(R);

            body.v[0] = V*body.x[1];
            body.v[1] = -V*body.x[0];
        }

        printf("min/max %.1f %.1f %.1f %.1f\n", minx, maxx, miny, maxy);
    }
};

template <typename T,tensor_flag flag>
void calc(const Config& c) {
    int n = c.get("nbody", "n", 512);
    int N = c.get("nbody", "N", 100000);
    int steps = c.get("nbody", "steps", 50000);
    double x0 = c.get("nbody", "x0", -10);
    double y0 = c.get("nbody", "y0", -10);
    double l = c.get("nbody", "l", 20);
    double dt = c.get("nbody", "dt", 0.001);
    double G = c.get("nbody", "G", 1);
    double vel = c.get("nbody", "vel", 4);
    int sgn = c.get("nbody", "sign", -1);
    int interval = c.get("plot","interval",100);

    NBody<T,true,flag> task(x0, y0, l, n, N, dt, G, vel, sgn);

    task.plot(0);

    for (int step = 0; step < steps; step++) {
        printf("step=%d\n", step);
        task.step();

        if ((step+1) % interval == 0) {
            task.plot(step+1);
        }
    }
}

int main(int argc, char** argv) {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    string datatype = c.get("solver", "datatype", "double");
    int periodic = c.get("nbody", "periodic", 1);

    if (datatype == "float") {
        if (periodic) {
            calc<float,tensor_flag::periodic>(c);
        } else {
            calc<float,tensor_flag::none>(c);
        }
    } else {
        if (periodic) {
            calc<double,tensor_flag::periodic>(c);
        } else {
            calc<double,tensor_flag::none>(c);
        }
    }

    return 0;
}
