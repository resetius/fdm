#include "ns_cyl.h"
#include "arpack_solver.h"
#include "velocity_plot.h"
#include "eigenvectors_storage.h"

using namespace std;
using namespace fdm;
using namespace asp;

template<typename T, bool check>
void calc(const Config& c) {
    using namespace std::chrono;

    using Task = NSCyl<T, check, tensor_flag::periodic>;
    using tensor = typename Task::tensor;
    Task ns(c);
    velocity_plotter<T,true, typename Task::tensor_flags> plot(
        ns.dr, ns.dz, ns.dphi,
        ns.nr, ns.nz, ns.nphi,
        ns.r0, ns.R,
        ns.h1, ns.h2,
        0, 2*M_PI, true);
    plot.set_labels("R", "Z", "PHI");

    int nev = c.get("test", "nev", 4);
    T tol = c.get("test", "tol", 1e-7);
    int maxit = c.get("test", "maxit", 10000);
    int rndresid = c.get("resid", "rnd", 1);
    T fixedresid = c.get("resid", "fixed", 1);
    T from = c.get("resid", "a", -1.0);
    T to = c.get("resid", "b", 1.0);

    const int steps = c.get("ns", "steps", 1);
    //const int plot_interval = c.get("plot", "interval", 100);
    //const int png = c.get("plot", "png", 1);
    //const int vtk = c.get("plot", "vtk", 0);
    int i, nphi, nz, nr;
    nphi = ns.nphi; nz = ns.nz; nr = ns.nr;
    tensor u{{0, nphi-1, 0, nz-1, 1, nr-1}};
    tensor v{{0, nphi-1, 0, nz-1, 1, nr}};
    tensor w{{0, nphi-1, 0, nz-1, 1, nr}};
    tensor p({0, nphi-1, 0, nz-1, 1, nr});

    int n = u.size+v.size+w.size+p.size;
    arpack_solver<T> solver(
        n, maxit,
        arpack_solver<T>::standard,
        arpack_solver<T>::largest_magnitude,
        rndresid == 1 ? arpack_solver<T>::random : arpack_solver<T>::fixed,
        tol);

    if (rndresid == 0) {
        solver.set_resid(fixedresid);
    } else if (rndresid == 2) {
        solver.set_resid_random(from, to);
    }

    auto t1 = steady_clock::now();
    int it = 0;

    vector<complex<T>> eigenvalues;
    vector<vector<T>> eigenvectors;

    for (int i = 0; i < nphi; i++) {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j <= nr; j++) {
                double r = ns.r0+ns.dr*j+ns.dr/2;
                ns.w0[i][k][j] =
                    -ns.U0*sq(ns.r0)/(sq(ns.R)-sq(ns.r0))
                    + ns.U0*sq(ns.r0)*sq(ns.R)/(sq(ns.R)-sq(ns.r0))/r/r;
            }
        }
    }
    int off = 0;

    // Мы решаем спектральную задачу для возмущения
    // На возмущение накладываются нулевые граничные условия
    ns.U0 = 0;

    printf("%d\n", n);
    auto t0 = steady_clock::now();
    solver.solve([&](T* y, const T* x) {
        // copy to inner domain
        auto t1 = steady_clock::now();

        memcpy(y, x, n*sizeof(T));
        off = 0;
        u.use(y + off); off += u.size;
        v.use(y + off); off += v.size;
        w.use(y + off); off += w.size;
        p.use(y + off); off += p.size;
        ns.u = u; ns.v = v; ns.w = w; ns.p = p;
        for (i = 0; i < steps; i++) {
            ns.L_step();
        }
        // copy out
        u = ns.u; v = ns.v; w = ns.w; p = ns.p;
        it++;
        auto t2 = steady_clock::now();
        auto interval1 = duration_cast<duration<double>>(t2 - t1);
        auto interval2 = duration_cast<duration<double>>(t1 - t0);
        t0 = t2;
        printf("Iter %d in %f/%f seconds (ns/arpack)\n",
               it, interval1.count(), interval2.count());
    }, eigenvalues, eigenvectors, nev);

    auto t2 = steady_clock::now();
    auto interval = duration_cast<duration<double>>(t2 - t1);
    printf("It took me '%f' seconds, '%d' iterations\n", interval.count(), it);

    int nconv = static_cast<int>(eigenvalues.size());
    vector<int> indices(nconv);
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&](int a, int b) {
        return abs(eigenvalues[a]) > abs(eigenvalues[b]);
    });

    int count = 0;
    for (int  i = 0; i < nconv; i++) {
        int j = indices[i];
        printf(" -> %.16e: %.16e %.16e \n",
               abs(eigenvalues[j]),
               eigenvalues[j].real(),
               eigenvalues[j].imag());
        if (abs(eigenvalues[j]) > 1) {
            count ++;
        }
    }
    printf("above 1: %d\n", count);

    if (count > 0) {
        plot.use(ns.u.vec, ns.v.vec, ns.w.vec);

        for (int  i = 0; i < count; i++) {
            int j = indices[i];
            u.use(&eigenvectors[j][off]); off += u.size;
            v.use(&eigenvectors[j][off]); off += v.size;
            w.use(&eigenvectors[j][off]); off += w.size;
            p.use(&eigenvectors[j][off]); off += p.size;

            ns.u = u; ns.v = v; ns.w = w; ns.p = p;

            plot.update();
            plot.plot(format("eigenvector_%07d.png", i), 0);
        }
    }

    if (count > 0) {
        string filename = format(
            "eigenvectors_%f_%d_%d_%d.nc",
            ns.Re, ns.nr, ns.nz, ns.nphi);
        eigenvectors_storage s(filename);
        s.save(eigenvectors, indices, c);
    }
}

template<typename T>
void calc1(const Config& c) {
    bool check = c.get("other", "check", 0) == 1;
    if (check) {
        calc<T, true>(c);
    } else {
        calc<T, false>(c);
    }
}

// Флетчер, том 2, страница 398
int main(int argc, char** argv) {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    string datatype = c.get("solver", "datatype", "double");

    if (datatype == "float") {
        using T = float;
        calc1<T>(c);
    } else {
        using T = double;
        calc1<T>(c);
    }

    return 0;
}
