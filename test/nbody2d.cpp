#include <random>
#include <list>
#include <thread>

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
using namespace asp;

template<typename T,bool check,tensor_flag flag,typename I=TSC2<T>>
class NBody {
public:
    using flags = typename short_flags<flag,flag>::value;
    using flags3 = typename short_flags<flag,flag,flag>::value;
    using tensor =  fdm::tensor<T,2,check,flags>;
    using tensor3 =  fdm::tensor<T,3,check,flags3>;

    T origin[2];
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

    struct Body {
        T x[2];
        T v[2];
        T a[2];
        T aprev[2];
        T mass;

        int k;
        int j;

        T F[2]; // Local force

        Body()
            : x{0}
            , v{0}
            , a{0}
            , aprev{0}
        { }
    };

    struct Cell {
        vector<int> bodies;
        vector<int> next;
    };

    vector<Body> bodies;
    int n0,n1,nn,nnn;
    tensor3 psi0,rhs;
    tensor psi,f;
    fdm::tensor<Cell,2,check,flags> cells;
    fdm::tensor<array<T,2>,2,check,flags> E;

    LaplRectFFT2<T,check,flags> solver;
    LaplCube<T,check,flags3> solver3;

    struct PlotTask {
        string fname;
        tensor f,psi;
        vector<Body> bodies;
    };

    concurrent_queue<PlotTask> q;
    std::thread thread;

    NBody(T x0, T y0, double l, int n, int N, double dt, double G, double vel, int sgn, int local, int pponly, int solar)
        : origin{x0, y0}
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
        , rcrit(2*h)
        , bodies(N)
        , n0(0)
        , n1(flag==tensor_flag::periodic?0:1)
        , nn(flag==tensor_flag::periodic?n-1:n)
        , nnn(flag==tensor_flag::periodic?n-1:n+1)

        , psi0({n1,nn,n1,nn,n1,nn})
        , rhs({n1,nn,n1,nn,n1,nn})

        , psi({n0,nnn,n0,nnn})
        , f({n0,nnn,n0,nnn})
        , cells({n0,nn,n0,nn})
        , E({n1,nn,n1,nn})
        , solver(h,h,l,l,n,n)
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
        move();
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
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                auto& bi = bodies[i];
                auto& bj = bodies[j];
                double eps = 0.00001;
                double R = 0;
                for (int k = 0; k < 2; k++) {
                    R += sq(bi.x[k]-bj.x[k]);
                }
                R = std::sqrt(R)+eps;
                for (int k = 0; k < 2; k++) {
                    bi.F[k] +=  sgn* bj.mass * G * (bi.x[k] - bj.x[k]) /R/R/R;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            auto& bi = bodies[i];
            double R = 0, RR = 0;
            for (int k = 0; k < 2; k++) {
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

            matrix_plotter plotter(matrix_plotter::settings()
                                   .sub(1, 2)
                                   .devname("pngcairo")
                                   .fname(task.fname));

            plenv(origin[1], origin[1]+l, origin[0], origin[0]+l, 1, 0);
            pllab("Y", "X", "");

            double r = 2*l / 2000.0;
            for (auto& body: task.bodies) {
                plarc(body.x[1], body.x[0], r, r, 0, 360, 0, 1);
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
        }
    }

    void calc_a_pp() {
        for (int i = 0; i < N; i++) {
            auto& bi = bodies[i];
            bi.a[0] = bi.a[1] = 0;

            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                auto& bj = bodies[j];
                double eps = 0.00001;
                double R = 0;
                for (int k = 0; k < 2; k++) {
                    R += sq(bi.x[k]-bj.x[k]);
                }
                R = std::sqrt(R);
                if (R < eps) continue;
                for (int k = 0; k < 2; k++) {
                    bi.a[k] +=  -bj.mass * G * (bi.x[k] - bj.x[k]) /R/R/R;
                }
            }
        }
    }

    void calc_a_pm() {
        // 1. mass to edges
#pragma omp parallel for
        for (int k = n0; k <= nnn; k++) {
            for (int j = n0; j <= nnn; j++) {
                f[k][j] = 0;
            }
        }

        I interpolator;

//#pragma omp parallel for // TODO: adjacent points edit
        for (auto& body : bodies) {
            double m = body.mass;

            int k0, j0;
            typename I::matrix M;

            interpolator.distribute(
                M,
                body.x[0]-origin[0],
                body.x[1]-origin[1],
                &j0, &k0, h);

            for (int k = 0; k < I::n; k++) {
                for (int j = 0; j < I::n; j++) {
                    if constexpr(flag == tensor_flag::periodic) {
                        f[k+k0][j+j0] += m * M[k][j];
                    } else {
                        if (k0+k >= n0 && k0+k <= nnn
                            && j0+j >= n0 && j0+j <= nnn)
                        {
                            f[k+k0][j+j0] += m * M[k][j];
                        }
                    }
                }
            }
        }

        // -4pi G ro
#pragma omp parallel for
        for (int k = n1; k <= nn; k++) {
            for (int j = n1; j <= nn; j++) {
                rhs[n/2][k][j] = 4*G*M_PI*f[k][j]/h/h/h;
            }
        }

        solver3.solve(&psi0[n1][n1][n1], &rhs[n1][n1][n1]);
        for (int k = n1; k <= nn; k++) {
            for (int j = n1; j <= nn; j++) {
                psi[k][j] = psi0[n/2][k][j];
            }
        }

        T beta=4./3.;
        for (int k = n1; k <= nn; k++) {
            for (int j = n1; j <= nn; j++){
                // 5.137, pp184, Hockney
                if constexpr(flag == tensor_flag::periodic) {
                    E[k][j][0] = -beta*(psi[k][j+1]-psi[k][j-1])/2/h
                        - (1-beta)*(psi[k][j+2]-psi[k][j-2])/4/h;
                    E[k][j][1] = -beta*(psi[k+1][j]-psi[k-1][j])/2/h
                        - (1-beta)*(psi[k+2][j]-psi[k-2][j])/4/h;
                } else {
                    if (j-2 >= n1 && j+2 <= nn && k-2 >= n1 && k+2 <= nn) {
                        E[k][j][0] = -beta*(psi[k][j+1]-psi[k][j-1])/2/h
                            - (1-beta)*(psi[k][j+2]-psi[k][j-2])/4/h;
                        E[k][j][1] = -beta*(psi[k+1][j]-psi[k-1][j])/2/h
                            - (1-beta)*(psi[k+2][j]-psi[k-2][j])/4/h;
                    } else {
                        E[k][j][0] = -(psi[k][j+1]-psi[k][j-1])/2/h;
                        E[k][j][1] = -(psi[k+1][j]-psi[k-1][j])/2/h;
                    }
                }
            }
        }

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

        if (local) {
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
    }

    void apply_bi_bj(Body& bi, Body& bj, bool apply_bj) {
        double eps = 0.0001;
        double R = 0;
        for (int k = 0; k < 2; k++) {
            R += sq(bi.x[k]-bj.x[k]);
        }
        R = std::sqrt(R); // +eps;

        if (R > eps && R < rcrit) {
            for (int k = 0; k < 2; k++) {
                bi.F[k] +=  -bj.mass * G * (bi.x[k] - bj.x[k]) /R/R/R
                    * (erfc( R/2/rcrit )
                       + R/rcrit/sqrt(M_PI)*exp(-R*R/4/rcrit/rcrit));
            }

            if (apply_bj) {
                for (int k = 0; k < 2; k++) {
                    bj.F[k] +=  -bi.mass * G * (bj.x[k] - bi.x[k]) /R/R/R
                    * (erfc( -(bj.x[k] - bi.x[k])/2/rcrit )
                       + (bj.x[k] - bi.x[k])/rcrit/sqrt(M_PI)*exp(-R*R/4/rcrit/rcrit));
                }
            }
        }
    }

    void calc_local_forces(Cell& cell) {
        int nbodies = static_cast<int>(cell.bodies.size());

        for (int i = 0; i < nbodies; i++) {
            for (int j = 0; j < nbodies; j++) {
                auto& bi = bodies[cell.bodies[i]];
                auto& bj = bodies[cell.bodies[j]];

                if (i!=j) {
                    apply_bi_bj(bi, bj, false);
                }
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
        I interpolator;

        for (auto& index : cell.bodies) {
            auto& body = bodies[index];

            double a[2] = {0};
            int k0, j0;
            typename I::matrix M;
            interpolator.distribute(
                M,
                body.x[0]-origin[0],
                body.x[1]-origin[1],
                &j0, &k0, h);


            for (int m = 0; m < 2; m++) {
                for (int k = 0; k < I::n; k++) {
                    for (int j = 0; j < I::n; j++) {
                        if constexpr(flag == tensor_flag::periodic) {
                            a[m] += E[k0+k][j0+j][m] * M[k][j];
                        } else {
                            if (k0+k >= n0 && k0+k <= nnn
                                && j0+j >= n0 && j0+j <= nnn)
                            {
                                a[m] += E[k0+k][j0+j][m] * M[k][j];
                            }
                        }
                    }
                }
            }

            for (int m = 0; m < 2; m++) {
                a[m] += body.F[m];
            }

            for (int m = 0; m < 2; m++) {
                body.a[m] = a[m];
            }
        }
    }

    void move() {
        for (int index = 0; index < N; index++) {
            auto& body =  bodies[index];
            // verlet integration
            // positions
            for (int m = 0; m < 2; m++) {
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
            for (int m = 0; m < 2; m++) {
                // velocity
                body.v[m] += 0.5 * dt * (body.a[m]+body.aprev[m]);
                // acceleration
                body.aprev[m] = body.a[m];
            }

            T x = body.x[0]-origin[0];
            T y = body.x[1]-origin[1];

            if (x < 0 || x > l) continue;
            if (y < 0 || y > l) continue;

            int next_j = floor(x / h);
            int next_k = floor(y / h);

            //verify(abs(body.k-next_k) <= 1, "Too fast, try decrease dt");
            //verify(abs(body.j-next_j) <= 1, "Too fast, try decrease dt");
            body.k = next_k; body.j = next_j;

            cells[body.k][body.j].next.push_back(index);
        }

#pragma omp parallel for
        for (int k = n0; k <= nn; k++) {
            for (int j = n0; j <= nn; j++) {
                cells[k][j].next.swap(cells[k][j].bodies);
            }
        }

        int k = 2;
        printf("%e %e %e %e \n", bodies[k].a[0],bodies[k].a[1], bodies[k].x[0],bodies[k].x[1]);
    }

    void init_points() {
        std::default_random_engine generator;
        std::uniform_real_distribution<T> distribution(0, l/2);


        T minx = 10000;
        T maxx = -10000;
        T miny = 10000;
        T maxy = -10000;

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
            }
        } else {

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

    }
};

template <typename T,tensor_flag flag, typename I>
void calc(const Config& c) {
    int n = c.get("nbody", "n", 512);
    int N = c.get("nbody", "N", 100000);
    int steps = c.get("nbody", "steps", 50000);
    double x0 = c.get("nbody", "x0", -10.0);
    double y0 = c.get("nbody", "y0", -10.0);
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

    NBody<T,true,flag,I> task(x0, y0, l, n, N, dt, G, vel, sgn, local, pponly, solar);

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
}

template<typename T,tensor_flag flag>
void calc1(const Config& c) {
    string interpolate = c.get("nbody", "interpolate", "tsc");
    if (interpolate == "tsc") {
        calc<T,flag,TSC2<T>>(c);
    } else if (interpolate == "pcs") {
        calc<T,flag,PCS2<T>>(c);
    }  else {
        calc<T,flag,CIC2<T>>(c);
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
