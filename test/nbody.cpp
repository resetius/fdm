#include <random>
#include <list>
#include <thread>
#include <chrono>

#include "config.h"
#include "tensor.h"
#include "blas.h"
#include "lapl_rect.h"
#include "lapl_cube.h"
#include "matrix_plot.h"
#include "asp_misc.h"
#include "concurrent_queue.h"
#include "interpolate.h"

using namespace fdm;
using namespace std;
using namespace std::chrono;
using namespace asp;

template<typename T,bool check,typename I=CIC3<T>>
class NBody {
public:
    static const tensor_flag flag = tensor_flag::periodic;
    using flags = typename short_flags<flag,flag,flag>::value;
    using tensor =  fdm::tensor<T,3,check,flags>;

    T origin[3];
    double l;
    int n, npp, N;
    double h, hpp;
    double dt;
    double G;
    double vel;
    int sgn;
    int local;
    int solar;
    double rcrit;
    double mass;

    struct Body {
        T x[3];
        T v[3];
        T a[3];
        T aprev[3];
        T mass;

        int i,k,j;
        int ipp,kpp,jpp;

        T F[3]; // Local force

        Body()
            : x{0}
            , v{0}
            , a{0}
            , aprev{0}
        { }
    };

    struct Cell {
        vector<int> bodies;
    };

    vector<Body> bodies;
    tensor rhs;
    tensor psi,f;
    fdm::tensor<Cell,3,check,flags> cells;
    fdm::tensor<Cell,3,check,flags> cellspp;
    fdm::tensor<array<T,3>,3,check,flags> E;

    LaplCube<T,check,flags> solver3;

    struct PlotTask {
        string fname;
        tensor f,psi;
        vector<Body> bodies;
    };

    concurrent_queue<PlotTask> q;
    std::thread thread;

    double distribute_time = 0;
    double poisson_time = 0;
    double diff_time = 0; // E time
    double local_p_time = 0;
    double accel_time = 0;
    double move_time = 0;

    NBody(T x0, T y0, T z0, double l, int n, int npp, int N, double dt, double G, double vel, int sgn, int local, int solar, T crit)
        : origin{x0, y0, z0}
        , l(l)
        , n(n) // число ячеек
        , npp(npp) // сетка для силы pp
        , N(N) // число тел
        , h(l/n)
        , hpp(l/npp)
        , dt(dt)
        , G(G)
        , vel(vel)
        , sgn(sgn)
        , local(local)
        , solar(solar)
        , rcrit(hpp*crit)
        , bodies(N)

        , rhs({0,n-1,0,n-1,0,n-1})

        , psi({0,n-1,0,n-1,0,n-1})
        , f({0,n-1,0,n-1,0,n-1})
        , cells({0,n-1,0,n-1,0,n-1})
        , cellspp({0,npp-1,0,npp-1,0,npp-1})

        , E({0,n-1,0,n-1,0,n-1})
        , solver3(h,h,h,l,l,l,n,n,n)
        , thread(plot_thread, l, origin, &q)
    {
        init_points();
    }

    ~NBody()
    {
        q.push(PlotTask{"", tensor({}), tensor({})});
        thread.join();
    }

    void step() {
        calc_a_pm();

        auto t1 = steady_clock::now();
        move();
        auto t2 = steady_clock::now();
        move_time += duration_cast<duration<double>>(t2 - t1).count();
    }

    void calc_error() {
        int n = static_cast<int>(bodies.size());

        calc_a_pm();

        for (int i = 0; i < n; i++) {
            bodies[i].F[0] = 0;
            bodies[i].F[1] = 0;
            bodies[i].F[2] = 0;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                auto& bi = bodies[i];
                auto& bj = bodies[j];
                double eps = 0.00001;
                double R = 0;
                for (int k = 0; k < 3; k++) {
                    R += sq(bi.x[k]-bj.x[k]);
                }
                R = std::sqrt(R)+eps;
                for (int k = 0; k < 3; k++) {
                    bi.F[k] +=  sgn* bj.mass * G * (bi.x[k] - bj.x[k]) /R/R/R;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            auto& bi = bodies[i];
            double R = 0, RR = 0;
            for (int k = 0; k < 3; k++) {
                R += sq(bi.a[k]-bi.F[k]);
                RR += sq(bi.F[k]);
            }
            R= std::sqrt(R); RR=std::sqrt(RR);
            R /= RR;
            //printf("< %e %e %e\n", bi.mass, bi.x[0], bi.x[1]);
            printf("> %e %+e %+e %+e %+e\n", R, bi.a[0], bi.F[0], bi.a[1], bi.F[1]);
        }
    }

    void plot(int step) {
        PlotTask task {format("step_%07d.png", step), f, psi, bodies};
        q.push(std::move(task));
    }

private:
    static void plot_thread(double l, T origin[2], concurrent_queue<PlotTask>* q) {
        while (true) {
            PlotTask task = q->pop();
            if (task.fname.empty()) {
                break;
            }
#if 0
            matrix_plotter plotter(matrix_plotter::settings()
                                   .sub(1, 2)
                                   .devname("pngcairo")
                                   .fname(task.fname));

            plenv(origin[1], origin[1]+l, origin[0], origin[0]+l, 1, 0);
            pllab("Y", "X", "");

            double r = 2*l / 2000.0;
            for (auto& body: task.bodies) {
                plarc(body.x[1], body.x[0], sqrt(body.mass)*r, sqrt(body.mass)*r, 0, 360, 0, 1);
            }
            /*
            plotter.plot(matrix_plotter::page()
                         .scalar(task.f)
                         .levels(10)
                         .tlabel("F")
                         .bounds(origin[0], origin[1], origin[0]+l, origin[1]+l));
            */
            plotter.plot(matrix_plotter::page()
                         .scalar(task.psi)
                         .levels(10)
                         .tlabel("Psi")
                         .bounds(origin[0], origin[1], origin[0]+l, origin[1]+l));

//            plflush();
//            plreplot();
//            printf("1\n");
#endif
        }
    }

    void distribute_mass(Body& body) {
        int i0, k0, j0;
        I interpolator;
        auto m = body.mass;
        typename I::matrix M;

        interpolator.distribute(
            M,
            body.x[0]-origin[0],
            body.x[1]-origin[1],
            body.x[2]-origin[2],
            &j0, &k0, &i0, h);
        for (int i = 0; i < I::n; i++) {
            for (int k = 0; k < I::n; k++) {
                for (int j = 0; j < I::n; j++) {
                    f[i+i0][k+k0][j+j0] += m * M[i][k][j];
                }
            }
        }
    }

    void distribute_masses() {
        // per cell

        for (int off = 0; off < I::n; off++) {
#pragma omp parallel for
            for (int i = off; i < n; i += I::n) {
                for (int k = off; k < n; k += I::n) {
                    for (int j = off; j < n; j += I::n) {
                        for (int index : cells[i][k][j].bodies) {
                            distribute_mass(bodies[index]);
                        }
                    }
                }
            }
        }
    }

    void calc_a_pm() {
        auto t1 = steady_clock::now();

        // 1. mass to edges
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    f[i][k][j] = - 4*G*M_PI*mass/l/l/l;
                }
            }
        }

        distribute_masses();

        auto t2 = steady_clock::now();

        distribute_time += duration_cast<duration<double>>(t2 - t1).count();

        // -4pi G ro
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    rhs[i][k][j] = 4*G*M_PI*f[i][k][j]/h/h/h;
                }
            }
        }

        solver3.solve(&psi[0][0][0], &rhs[0][0][0]);

        auto t3 = steady_clock::now();
        poisson_time += duration_cast<duration<double>>(t3 - t2).count();

        T beta=4./3.;
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++){
                    // 5.137, pp184, Hockney
                    E[i][k][j][0] = -beta*(psi[i][k][j+1]-psi[i][k][j-1])/2/h
                        - (1-beta)*(psi[i][k][j+2]-psi[i][k][j-2])/4/h;
                    E[i][k][j][1] = -beta*(psi[i][k+1][j]-psi[i][k-1][j])/2/h
                        - (1-beta)*(psi[i][k+2][j]-psi[i][k-2][j])/4/h;
                    E[i][k][j][2] = -beta*(psi[i+1][k][j]-psi[i-1][k][j])/2/h
                        - (1-beta)*(psi[i+2][k][j]-psi[i-2][k][j])/4/h;
                }
            }
        }

        auto t4 = steady_clock::now();
        diff_time += duration_cast<duration<double>>(t4 - t3).count();

#pragma omp parallel for
        for (auto& body : bodies) {
            for (int i = 0; i < 3; i++) {
                body.F[i] = 0;
            }
        }

        if (local) {
#pragma omp parallel for
            for (int i = 0; i < npp; i++) {
                for (int k = 0; k < npp; k++) {
                    for (int j = 0; j < npp; j++) {
                        auto& cell = cellspp[i][k][j];

                        for (int i0 = -1; i0 <= 1; i0++) {
                            for (int k0 = -1; k0 <= 1; k0++) {
                                for (int j0 = -1; j0 <= 1; j0++) {
                                    T off[] = {0.0,0.0,0.0};

                                    if (k+k0 < 0)  off[0] = -l;
                                    if (k+k0 >= n) off[0] =  l;
                                    if (j+j0 < 0)  off[1] = -l;
                                    if (j+j0 >= n) off[1] =  l;
                                    if (i+i0 < 0)  off[2] = -l;
                                    if (i+i0 >= n) off[2] =  l;
                                    calc_local_forces(cell, cellspp[i+i0][k+k0][j+j0], off);
                                }
                            }
                        }
                    }
                }
            }
        }

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < n; j++) {
                    cells[i][k][j].bodies.clear();
                }
            }
        }
#pragma omp parallel for
        for (int i = 0; i < npp; i++) {
            for (int k = 0; k < npp; k++) {
                for (int j = 0; j < npp; j++) {
                    cellspp[i][k][j].bodies.clear();
                }
            }
        }

        auto t5 = steady_clock::now();
        local_p_time += duration_cast<duration<double>>(t5 - t4).count();

        calc_accelerations();

        auto t6 = steady_clock::now();
        accel_time += duration_cast<duration<double>>(t6 - t5).count();
    }

    void apply_bi_bj(Body& bi, Body& bj, T off[2]) {
        double R = 0;
        for (int k = 0; k < 3; k++) {
            R += sq(bi.x[k]-(bj.x[k]+off[k]));
        }
        R = std::sqrt(R) + 0.0001;

        if (R < rcrit) {
            for (int k = 0; k < 3; k++) {
                bi.F[k] +=  -bj.mass * G * (bi.x[k] - (bj.x[k]+off[k])) /R/R/R
                    * (erfc( R/2/rcrit )
                       + R/rcrit/sqrt(M_PI)*exp(-R*R/4/rcrit/rcrit));
            }
        }
    }

    void calc_local_forces(Cell& cell, Cell& other, T off[2]) {
        for (auto& bi : cell.bodies) {
            for (auto& bj : other.bodies) {
                if (bi != bj) {
                    apply_bi_bj(bodies[bi], bodies[bj], off);
                }
            }
        }
    }

    void calc_accelerations() {
        I interpolator;

#pragma omp parallel for
        for (auto& body : bodies) {
            int i0, k0, j0;
            typename I::matrix M;
            interpolator.distribute(
                M,
                body.x[0]-origin[0],
                body.x[1]-origin[1],
                body.x[2]-origin[2],
                &j0, &k0, &i0, h);

            for (int m = 0; m < 3; m++) {
                body.a[m] = 0;
            }

            // local pp-force
            for (int m = 0; m < 3; m++) {
                body.a[m] += body.F[m];
            }

            // pm-force
            for (int i = 0; i < I::n; i++) {
                for (int k = 0; k < I::n; k++) {
                    for (int j = 0; j < I::n; j++) {
                        for (int m = 0; m < 3; m++) {
                            body.a[m] += E[i0+i][k0+k][j0+j][m] * M[i][k][j];
                        }
                    }
                }
            }
        }
    }

    void move() {
#pragma omp parallel for
        for (auto& body : bodies) {

            // verlet integration
            // positions
            for (int m = 0; m < 3; m++) {
                body.x[m] += dt * body.v[m] + 0.5 * dt * dt * body.aprev[m];

                if (body.x[m] < origin[m]) {
                    body.x[m] += l;
                }
                if (body.x[m] >= origin[m]+l) {
                    body.x[m] -= l;
                }
            }
            for (int m = 0; m < 3; m++) {
                // velocity
                body.v[m] += 0.5 * dt * (body.a[m]+body.aprev[m]);
                // acceleration
                body.aprev[m] = body.a[m];
            }

            body.j = floor((body.x[0]-origin[0]) / h);
            body.k = floor((body.x[1]-origin[1]) / h);
            body.i = floor((body.x[2]-origin[2]) / h);

            body.jpp = floor((body.x[0]-origin[0]) / hpp);
            body.kpp = floor((body.x[1]-origin[1]) / hpp);
            body.ipp = floor((body.x[2]-origin[2]) / hpp);
        }

        // TODO
        for (int index = 0; index < N; index++) {
            auto& body = bodies[index];
            cells[body.i][body.k][body.j].bodies.push_back(index);
            cellspp[body.ipp][body.kpp][body.jpp].bodies.push_back(index);
        }

        int k = 2;
        printf("%e %e %e %e \n",
               bodies[k].a[0],bodies[k].a[1], bodies[k].x[0],bodies[k].x[1]);
    }

    void init_points() {
        mass = 0;
        if (solar) {
            G = 2.96e-6;
            struct Abc {
                T x[3];
                T v[3];
                T mass;
                int fixed;
                const char* name;
            } abc[] = {
                {{0, 0, 0}, {0, 0, 0}, 333333, 1, "Sun"},
                {{0, 0.39, 0}, {1.58, 0, 0}, 0.038, 0, "Mercury"},
                {{0, 0.72, 0}, {1.17, 0, 0}, 0.82, 0, "Venus"},
                {{0, 1, 0}, {1, 0, 0}, 1, 0, "Earth"},
                {{0, 1.00256, 0}, {1.03, 0, 0}, 0.012, 0, "Moon"},
                {{0, 1.51, 0}, {0.8, 0, 0}, 0.1, 0, "Mars"},
                {{0, 5.2, 0}, {0.43, 0, 0}, 317, 0, "Jupiter"},
                {{0, 9.3, 0}, {0.32, 0, 0}, 95, 0, "Saturn"},
                {{0, 19.3, 0}, {0.23, 0, 0}, 14.5, 0, "Uranus"},
                {{0, 30, 0}, {0.18, 0, 0}, 16.7, 0, "Neptune"}
            };

            bodies.resize(10);
            N = bodies.size();
            for (int index = 0; index < static_cast<int>(bodies.size()); index++) {
                auto& body = bodies[index];
                body.x[0] = abc[index].x[0];
                body.x[1] = abc[index].x[1];
                body.v[0] = abc[index].v[0];
                body.v[1] = abc[index].v[1];
                body.mass = abc[index].mass;

                mass += body.mass;
            }
        } else {
            std::default_random_engine generator;
            std::uniform_real_distribution<T> distribution(0.0, 1.0);

            T minx = 10000;
            T maxx = -10000;
            T miny = 10000;
            T maxy = -10000;

            for (int index = 0; index < static_cast<int>(bodies.size()); index++) {
                auto& body = bodies[index];
                for (int i = 0; i < 3; i++) {
                    //body.x[i] = l*distribution(generator)/11 + origin[i] + 5*l/11;
                    body.x[i] = l*distribution(generator) + origin[i];
                }
                body.mass = 0.2 + 1.5 * distribution(generator);

                body.j = floor((body.x[0]-origin[0]) / h);
                body.k = floor((body.x[1]-origin[1]) / h);
                body.i = floor((body.x[2]-origin[2]) / h);

                body.jpp = floor((body.x[0]-origin[0]) / hpp);
                body.kpp = floor((body.x[1]-origin[1]) / hpp);
                body.ipp = floor((body.x[2]-origin[2]) / hpp);

                cells[body.i][body.k][body.j].bodies.push_back(index);
                cellspp[body.ipp][body.kpp][body.jpp].bodies.push_back(index);

                minx = min(minx, body.x[0]);
                maxx = max(maxx, body.x[0]);
                miny = min(miny, body.x[1]);
                maxy = max(maxy, body.x[1]);

                double R = std::sqrt(blas::dot(3, body.x, 1, body.x, 1));
                double V = vel/sqrt(R);

                body.v[0] = V*body.x[1];
                body.v[1] = -V*body.x[0];

                mass += body.mass;
            }

            printf("min/max %.1f %.1f %.1f %.1f\n", minx, maxx, miny, maxy);
        }

    }
};

template <typename T, bool check, typename I>
void calc(const Config& c) {
    int n = c.get("nbody", "n", 32);
    int npp = c.get("nbody", "npp", 64);
    int N = c.get("nbody", "N", 100000);
    int steps = c.get("nbody", "steps", 50000);
    double x0 = c.get("nbody", "x0", -10.0);
    double y0 = c.get("nbody", "y0", -10.0);
    double z0 = c.get("nbody", "z0", -10.0);
    double l = c.get("nbody", "l", 20.0);
    double dt = c.get("nbody", "dt", 0.001);
    double G = c.get("nbody", "G", 1.0);
    double vel = c.get("nbody", "vel", 4.0);
    int sgn = c.get("nbody", "sign", -1);
    int interval = c.get("plot","interval",100);
    int local = c.get("nbody", "local", 0); // need to check
    int solar = c.get("nbody", "solar", 0);
    int error = c.get("nbody", "error", 0);
    double crit = c.get("nbody", "rcrit", 1.0);

    NBody<T,check,I> task(x0, y0, z0, l, n, npp, N, dt, G, vel, sgn, local, solar, crit);

    if (error) {
        task.calc_error();
        return;
    }

    task.plot(0);

    for (int step = 0; step < steps; step++) {
        printf("step=%d\n", step);
        task.step();

        if ((step+1) % interval == 0) {
            task.plot(step+1);
        }
    }

    printf("Stat Total:\n");
    printf("distribute_time: %e\n", task.distribute_time);
    printf("poisson_time: %e\n", task.poisson_time);
    printf("diff_time: %e\n", task.diff_time);
    printf("local_p_time: %e\n", task.local_p_time);
    printf("accel_time: %e\n", task.accel_time);
    printf("move_time: %e\n", task.move_time);

    printf("Stat Per Step:\n");
    printf("distribute_time: %.2fms\n", 1000.0*task.distribute_time / steps);
    printf("poisson_time: %.2fms\n", 1000.0*task.poisson_time / steps);
    printf("diff_time: %.2fms\n", 1000.0*task.diff_time / steps);
    printf("local_p_time: %.2fms\n", 1000.0*task.local_p_time / steps);
    printf("accel_time: %.2fms\n", 1000.0*task.accel_time / steps);
    printf("move_time: %.2fms\n", 1000.0*task.move_time / steps);
    printf("total: %.2fms\n",
           1000.0*(task.distribute_time+
                   task.poisson_time+
                   task.diff_time+
                   task.local_p_time+
                   task.accel_time+
                   task.move_time) / steps);
}

template<typename T, typename I>
void calc2(const Config& c) {
    bool check = c.get("nbody", "check", 1);
    if (check) {
        calc<T,true,I>(c);
    } else {
        calc<T,false,I>(c);
    }
}

template<typename T>
void calc1(const Config& c) {
    string interpolate = c.get("nbody", "interpolate", "tsc");
    calc2<T,CIC3<T>>(c);
}

int main(int argc, char** argv) {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    string datatype = c.get("solver", "datatype", "double");

    if (datatype == "float") {
        calc1<float>(c);
    } else {
        calc1<double>(c);
    }

    return 0;
}
