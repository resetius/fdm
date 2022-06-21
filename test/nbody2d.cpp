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

    struct Body {
        T x[2];
        T v[2];
        T mass;
    };

    vector<Body> bodies;
    int n0,n1,nn,nnn;
    tensor psi, rhs, f;

    LaplRectFFT2<T,check,flags> solver;

    NBody(double x0, double y0, double l, int n, int N)
        : origin{x0, y0}
        , l(l)
        , n(n) // n+1 - число ячеек (для сдвинутых сеток n - число ячеек)
        , N(N) // число тел
        , h(l / (n+1))
        , dt(0.001)
        , bodies(N)
        , n0(0)
        , n1(flag==tensor_flag::periodic?0:1)
        , nn(flag==tensor_flag::periodic?n-1:n)
        , nnn(flag==tensor_flag::periodic?n-1:n+1)
        , psi({n1,nn,n1,nn})
        , rhs({n1,nn,n1,nn})
        , f({n0,nnn,n0,nnn})
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

        const double G = 2;

        // -4pi G ro
#pragma omp parallel for
        for (int k = n1; k <= nn; k++) {
            for (int j = n1; j <= nn; j++) {
                rhs[k][j] = -4.*G*M_PI*f[k][j];
                // zero bound
            }
        }
        solver.solve(&psi[1][1], &rhs[1][1]);

#pragma omp parallel for
        for (auto& body : bodies) {
            int k,j;

            double x = body.x[0]-origin[0];
            double y = body.x[1]-origin[1];

            if (x < 0 || x > l) continue;
            if (y < 0 || y > l) continue;

            j = floor(x / h);
            k = floor(y / h);

            if constexpr(flag == tensor_flag::none) {
                if (k == 0 || j == 0 || k == n || j == n) {
                    continue;
                }
            }

            double a[2];
            a[0] = (psi[k][j+1]-psi[k][j-1])/2/h;
            a[1] = (psi[k+1][j]-psi[k-1][j])/2/h;

            for (int m = 0; m < 2; m++) {
                body.v[m] += dt * a[m];
                body.x[m] += dt * body.v[m];

                if constexpr(flag == tensor_flag::periodic) {
                    if (body.x[m] < origin[m]) {
                        body.x[m] += l;
                    }
                    if (body.x[m] >= origin[m]+l) {
                        body.x[m] -= l;
                    }
                }
            }
        }

        int k = 10;
        printf("%e %e \n", bodies[k].x[0],bodies[k].x[1]);
    }

private:
    void init_points() {
        std::default_random_engine generator;
        std::uniform_real_distribution<T> distribution(0, l/2);

        T minx = 10000;
        T maxx = -10000;
        T miny = 10000;
        T maxy = -10000;

        for (auto& body : bodies) {
            for (int i = 0; i < 2; i++) {
                body.x[i] = distribution(generator) + origin[i] + l/4;
            }
            body.mass = 0.2 + 1.5 * distribution(generator) / l;

            minx = min(minx, body.x[0]);
            maxx = max(maxx, body.x[0]);
            miny = min(miny, body.x[1]);
            maxy = max(maxy, body.x[1]);

            double R = std::sqrt(blas::dot(3, body.x, 1, body.x, 1));
            //double V = sqrt(15)/sqrt(R);
            double V = sqrt(100)/sqrt(R);

            body.v[0] = V*body.x[1];
            body.v[1] = -V*body.x[0];
        }

        printf("min/max %.1f %.1f %.1f %.1f\n", minx, maxx, miny, maxy);
    }
};

template <typename T>
void calc(const Config& c) {
    int n = 512;
    NBody<T,true,tensor_flag::periodic> task(-10, -10, 20, n, 100000);

    for (int step = 0; step < 50000; step++) {
        printf("step=%d\n", step);
        task.step();

        if (step % 100 == 0) {
        string fname = format("step_%07d.png", step);
        matrix_plotter plotter(matrix_plotter::settings()
                               .sub(1, 2)
                               .devname("pngcairo")
                               .fname(fname));

        plotter.plot(matrix_plotter::page()
                     .scalar(task.f)
                     .levels(10)
                     .tlabel("F")
                     .bounds(-10, -10, 10, 10));

        plotter.plot(matrix_plotter::page()
                     .scalar(task.psi)
                     .levels(10)
                     .tlabel("Psi")
                     .bounds(-10, -10, 10, 10));
        }
    }

    FILE* f = fopen("dump.bin", "wb");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f, "%.16f ", task.f[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);

}

int main(int argc, char** argv) {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    string datatype = c.get("solver", "datatype", "double");

    if (datatype == "float") {
        calc<float>(c);
    } else {
        calc<double>(c);
    }

    return 0;
}
