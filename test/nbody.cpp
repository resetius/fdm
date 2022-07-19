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

template<typename T,bool check,tensor_flag flag,typename I=CIC3<T>>
class NBody {
public:
    using flags = typename short_flags<flag,flag,flag>::value;
    using tensor =  fdm::tensor<T,3,check,flags>;

    T origin[3];
    double l;
    int n, N;
    double h;
    double dt;
    double G;
    double vel;
    int sgn;
    int local;
    int pponly;
    int solar;
    double rcrit;
    double rsoft;
    double mass;

    struct Body {
        T x[3];
        T v[3];
        T a[3];
        T aprev[3];
        T mass;
        int enabled;

        int i;
        int k;
        int j;

        T F[3]; // Local force

        Body()
            : x{0}
            , v{0}
            , a{0}
            , aprev{0}
            , enabled(true)
        { }
    };

    struct Cell {
        vector<int> bodies;
        bool distributed = false;
    };

    vector<Body> bodies;
    int n0,n1,nn,nnn;
    tensor rhs;
    tensor psi,f;
    fdm::tensor<Cell,3,check,flags> cells;
    fdm::tensor<array<T,3>,3,check,flags> E;

    LaplCube<T,check,flags> solver3;

    struct PlotTask {
        string fname;
        tensor f,psi;
        vector<Body> bodies;
    };

    concurrent_queue<PlotTask> q;
    std::thread thread;

    int collisions = 0;
    double distribute_time = 0;
    double poisson_time = 0;
    double diff_time = 0; // E time
    double local_p_time = 0;
    double accel_time = 0;
    double move_time = 0;

    NBody(T x0, T y0, T z0, double l, int n, int N, double dt, double G, double vel, int sgn, int local, int pponly, int solar, T crit, T rsoft)
        : origin{x0, y0, z0}
        , l(l)
        , n(n) // n+1 - число ячеек (для сдвинутых сеток n - число ячеек)
        , N(N) // число тел
        , h(l / (n+1))
        , dt(dt)
        , G(G)
        , vel(vel)
        , sgn(sgn)
        , local(local)
        , pponly(pponly)
        , solar(solar)
        , rcrit(h*crit)
        , rsoft(h*rsoft)
        , bodies(N)
        , n0(0)
        , n1(flag==tensor_flag::periodic?0:1)
        , nn(flag==tensor_flag::periodic?n-1:n)
        , nnn(flag==tensor_flag::periodic?n-1:n+1)

        , rhs({n1,nn,n1,nn,n1,nn})

        , psi({n0,nnn,n0,nnn,n0,nnn})
        , f({n0,nnn,n0,nnn,n0,nnn})
        , cells({n0,nn,n0,nn,n0,nn})
        , E({n1,nn,n1,nn,n1,nn})
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
        if (pponly) {
            calc_a_pp();
        } else {
            calc_a_pm();
        }

        auto t1 = steady_clock::now();
        move();
        auto t2 = steady_clock::now();
        move_time += duration_cast<duration<double>>(t2 - t1).count();
    }

    void calc_error() {
        int n = static_cast<int>(bodies.size());

        if (pponly) {
            calc_a_pp();
        } else {
            calc_a_pm();
        }

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
                if (body.enabled) {
                    plarc(body.x[1], body.x[0], sqrt(body.mass)*r, sqrt(body.mass)*r, 0, 360, 0, 1);
                }
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

    void join_bodies(Body& bi, Body& bj) {
        collisions ++;
        for (int k = 0; k < 3; k++) {
            bi.v[k] = (bi.mass*bi.v[k]+bj.mass*bj.v[k]) / (bi.mass + bj.mass);
        }
        bi.mass = (bi.mass + bj.mass);
        bj.mass = 0;
        bj.enabled = false;
    }

    void calc_a_pp() {
        for (int i = 0; i < N; i++) {
            auto& bi = bodies[i];
            bi.a[0] = bi.a[1] = bi.a[2] = 0;

            if (!bi.enabled) continue;

            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                auto& bj = bodies[j];
                if (!bj.enabled) continue;

                double R = 0;
                for (int k = 0; k < 3; k++) {
                    R += sq(bi.x[k]-bj.x[k]);
                }
                R = std::sqrt(R);
                if (R < rsoft) {
                    join_bodies(bi, bj);
                    continue;
                }
                for (int k = 0; k < 3; k++) {
                    bi.a[k] +=  -bj.mass * G * (bi.x[k] - bj.x[k]) /R/R/R;
                }
            }
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
                    if constexpr(flag == tensor_flag::periodic) {
                        f[i+i0][k+k0][j+j0] += m * M[i][k][j];
                    } else {
                        if (i0+i >= n0 && i0+i <= nnn
                            && k0+k >= n0 && k0+k <= nnn
                            && j0+j >= n0 && j0+j <= nnn)
                        {
                            f[i+i0][k+k0][j+j0] += m * M[i][k][j];
                        }
                    }
                }
            }
        }
    }

    void distribute_masses() {
        // per cell
        for (int i = n0; i <= nn; i ++) {
            for (int k = n0; k <= nn; k ++) {
                for (int j = n0; j <= nn; j ++) {
                    cells[i][k][j].distributed = false;
                }
            }
        }

        for (int off = 0; off < I::n; off++) {
#pragma omp parallel for
            for (int i = n0+off; i <= nn; i += I::n) {
                for (int k = n0+off; k <= nn; k += I::n) {
                    for (int j = n0+off; j <= nn; j += I::n) {
                        verify(!cells[i][k][j].distributed);

                        for (int index : cells[i][k][j].bodies) {
                            distribute_mass(bodies[index]);
                        }

                        cells[i][k][j].distributed = true;
                    }
                }
            }
        }

        if constexpr(check==true) {
            for (int i = n0; i <= nn; i += I::n) {
                for (int k = n0; k <= nn; k += I::n) {
                    for (int j = n0; j <= nn; j += I::n) {
                        verify(cells[i][k][j].distributed);
                    }
                }
            }
        }
    }

    void calc_a_pm() {
        auto t1 = steady_clock::now();

        // 1. mass to edges
#pragma omp parallel for
        for (int i = n0; i <= nnn; i++) {
            for (int k = n0; k <= nnn; k++) {
                for (int j = n0; j <= nnn; j++) {
                    if constexpr(flag == tensor_flag::periodic) {
                        f[i][k][j] = - 4*G*M_PI*mass/l/l/l;
                    } else {
                        f[i][k][j] = 0;
                    }
                }
            }
        }

        distribute_masses();

        auto t2 = steady_clock::now();

        distribute_time += duration_cast<duration<double>>(t2 - t1).count();

        // -4pi G ro
#pragma omp parallel for
        for (int i = n1; i <= nn; i++) {
            for (int k = n1; k <= nn; k++) {
                for (int j = n1; j <= nn; j++) {
                    rhs[i][k][j] = 4*G*M_PI*f[i][k][j]/h/h/h;
                }
            }
        }

        solver3.solve(&psi[n1][n1][n1], &rhs[n1][n1][n1]);

        auto t3 = steady_clock::now();
        poisson_time += duration_cast<duration<double>>(t3 - t2).count();

        T beta=4./3.;
#pragma omp parallel for
        for (int i = n1; i <= nn; i++) {
            for (int k = n1; k <= nn; k++) {
                for (int j = n1; j <= nn; j++){
                    // 5.137, pp184, Hockney
                    if constexpr(flag == tensor_flag::periodic) {
                        E[i][k][j][0] = -beta*(psi[i][k][j+1]-psi[i][k][j-1])/2/h
                            - (1-beta)*(psi[i][k][j+2]-psi[i][k][j-2])/4/h;
                        E[i][k][j][1] = -beta*(psi[i][k+1][j]-psi[i][k-1][j])/2/h
                            - (1-beta)*(psi[i][k+2][j]-psi[i][k-2][j])/4/h;
                        E[i][k][j][2] = -beta*(psi[i+1][k][j]-psi[i-1][k][j])/2/h
                            - (1-beta)*(psi[i+2][k][j]-psi[i-2][k][j])/4/h;
                    } else {
                        if (j-2 >= n1 && j+2 <= nn && k-2 >= n1 && k+2 <= nn) {
                            E[i][k][j][0] = -beta*(psi[i][k][j+1]-psi[i][k][j-1])/2/h
                                - (1-beta)*(psi[i][k][j+2]-psi[i][k][j-2])/4/h;
                            E[i][k][j][1] = -beta*(psi[i][k+1][j]-psi[i][k-1][j])/2/h
                                - (1-beta)*(psi[i][k+2][j]-psi[i][k-2][j])/4/h;
                            E[i][k][j][2] = -beta*(psi[i+1][k][j]-psi[i-1][k][j])/2/h
                                - (1-beta)*(psi[i-2][k][j]-psi[i-2][k][j])/4/h;
                        } else {
                            E[i][k][j][0] = -(psi[i][k][j+1]-psi[i][k][j-1])/2/h;
                            E[i][k][j][1] = -(psi[i][k+1][j]-psi[i][k-1][j])/2/h;
                            E[i][k][j][2] = -(psi[i+1][k][j]-psi[i-1][k][j])/2/h;
                        }
                    }
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
            for (int i = n0; i <= nn; i++) {
                for (int k = n0; k <= nn; k++) {
                    for (int j = n0; j <= nn; j++) {
                        auto& cell = cells[i][k][j];

                        for (int i0 = -1; i0 <= 1; i0++) {
                            for (int k0 = -1; k0 <= 1; k0++) {
                                for (int j0 = -1; j0 <= 1; j0++) {
                                    T off[] = {0.0,0.0,0.0};

                                    if constexpr(flag == tensor_flag::periodic) {
                                        if (k+k0 < 0)  off[0] = -l;
                                        if (k+k0 > nn) off[0] =  l;
                                        if (j+j0 < 0)  off[1] = -l;
                                        if (j+j0 > nn) off[1] =  l;
                                        if (i+i0 < 0)  off[2] = -l;
                                        if (i+i0 > nn) off[2] =  l;
                                        calc_local_forces(cell, cells[i+i0][k+k0][j+j0], off);
                                    } else {
                                        if (k+k0 >= 0 && k+k0 <= nn && j+j0 >=0 && j+j0 <= nn && i+i0 >= 0 && i+i0 <= nn) {
                                            calc_local_forces(cell, cells[i+i0][k+k0][j+j0], off);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        #pragma omp parallel for
        for (int i = n0; i <= nn; i++) {
            for (int k = n0; k <= nn; k++) {
                for (int j = n0; j <= nn; j++) {
                    cells[i][k][j].bodies.clear();
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
        if (!bi.enabled || !bj.enabled) return;

        double R = 0;
        for (int k = 0; k < 3; k++) {
            R += sq(bi.x[k]-(bj.x[k]+off[k]));
        }
        R = std::sqrt(R); // +eps;

        if (R < rcrit) {
            if (R < rsoft) {
                join_bodies(bi, bj);
                return;
            }

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
                            if constexpr(flag == tensor_flag::periodic) {
                                body.a[m] += E[i0+i][k0+k][j0+j][m] * M[i][k][j];
                            } else {
                                if (k0+k >= n0 && k0+k <= nnn
                                    && j0+j >= n0 && j0+j <= nnn
                                    && i0+i >= n0 && i0+i <= nnn)
                                {
                                    body.a[m] += E[i0+i][k0+k][j0+j][m] * M[i][k][j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void move() {
#pragma omp parallel for
        for (auto& body : bodies) {
            if (!body.enabled) continue;

            // verlet integration
            // positions
            for (int m = 0; m < 3; m++) {
                body.x[m] += dt * body.v[m] + 0.5 * dt * dt * body.aprev[m];

                if constexpr(flag == tensor_flag::periodic) {
                    if (body.x[m] < origin[m]) {
                        body.x[m] += l;
                    }
                    if (body.x[m] >= origin[m]+l) {
                        body.x[m] -= l;
                    }
                }
            }
            for (int m = 0; m < 3; m++) {
                // velocity
                body.v[m] += 0.5 * dt * (body.a[m]+body.aprev[m]);
                // acceleration
                body.aprev[m] = body.a[m];
            }

            T x = body.x[0]-origin[0];
            T y = body.x[1]-origin[1];
            T z = body.x[2]-origin[2];

            if constexpr(flag == tensor_flag::none) {
                if (x < 0 || x > l) continue;
                if (y < 0 || y > l) continue;
                if (z < 0 || z > l) continue;
            }

            int next_j = floor(x / h);
            int next_k = floor(y / h);
            int next_i = floor(z / h);

            //verify(abs(body.k-next_k) <= 1, "Too fast, try decrease dt");
            //verify(abs(body.j-next_j) <= 1, "Too fast, try decrease dt");
            body.i = next_i; body.k = next_k; body.j = next_j;
        }

        // TODO
        for (int index = 0; index < N; index++) {
            auto& body = bodies[index];
            if (!body.enabled) continue;
            cells[body.i][body.k][body.j].bodies.push_back(index);
        }

        int k = 2;
        printf("%e %e %e %e %d \n",
               bodies[k].a[0],bodies[k].a[1], bodies[k].x[0],bodies[k].x[1],
               collisions);
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

                cells[body.i][body.k][body.j].bodies.push_back(index);

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

template <typename T,tensor_flag flag, bool check, typename I>
void calc(const Config& c) {
    int n = c.get("nbody", "n", 32);
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
    int pponly = c.get("nbody", "pponly", 0);
    int solar = c.get("nbody", "solar", 0);
    int error = c.get("nbody", "error", 0);
    double crit = c.get("nbody", "rcrit", l/n);
    double rsoft = c.get("nbody", "rsoft", 0.01);

    NBody<T,check,flag,I> task(x0, y0, z0, l, n, N, dt, G, vel, sgn, local, pponly, solar, crit, rsoft);

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

template<typename T,tensor_flag flag, typename I>
void calc2(const Config& c) {
    bool check = c.get("nbody", "check", 1);
    if (check) {
        calc<T,flag,true,I>(c);
    } else {
        calc<T,flag,false,I>(c);
    }
}

template<typename T,tensor_flag flag>
void calc1(const Config& c) {
    string interpolate = c.get("nbody", "interpolate", "tsc");
    calc2<T,flag,CIC3<T>>(c);
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
            calc1<float,tensor_flag::periodic>(c);
        } else {
            calc1<float,tensor_flag::none>(c);
        }
    } else {
        if (periodic) {
            calc1<double,tensor_flag::periodic>(c);
        } else {
            calc1<double,tensor_flag::none>(c);
        }
    }

    return 0;
}
