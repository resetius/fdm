#include "ns_cyl.h"

#include "arpack_solver.h"
#include "umfpack_solver.h"
#include "gmres_solver.h"
#include "superlu_solver.h"
#include "jacobi_solver.h"

template<typename T, template<typename> class Solver, bool check>
void calc(const Config& c) {
    using namespace std::chrono;

    using Task = NSCyl<T, Solver, check>;
    using tensor = typename Task::tensor;
    Task ns(c);

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
    tensor u{{0, nphi-1, 1, nz,   1, nr-1}};
    tensor v{{0, nphi-1, 1, nz-1, 1, nr}};
    tensor w{{0, nphi-1, 1, nz,   1, nr}};
    tensor p({0, nphi-1, 1, nz,   1, nr});

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

    solver.solve([&](T* y, const T* x) {
        // copy to inner domain
        memcpy(y, x, n*sizeof(T));
        int off = 0;
        u.vec = y + off; off += u.size;
        v.vec = y + off; off += v.size;
        w.vec = y + off; off += w.size;
        p.vec = y + off; off += p.size;
        ns.u = u; ns.v = v; ns.w = w; ns.p = p;
        for (i = 0; i < steps; i++) {
            ns.step();
        }
        // copy out
        u = ns.u; v = ns.v; w = ns.w; p = ns.p;
        it++;
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

    for (int  i = 0; i < nconv; i++) {
        int j = indices[i];
        printf(" -> %e %e \n",
               eigenvalues[j].real(), eigenvalues[j].imag());
    }
}

template<typename T, template<typename> class Solver>
void calc1(const Config& c) {
    bool check = c.get("other", "check", 0) == 1;
    if (check) {
        calc<T, Solver, true>(c);
    } else {
        calc<T, Solver, false>(c);
    }
}

// Флетчер, том 2, страница 398
int main(int argc, char** argv) {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    string solver = c.get("solver", "name", "umfpack");
    string datatype = c.get("solver", "datatype", "double");

    if (datatype == "float") {
        using T = float;
        if (solver == "gmres") {
            calc1<T, gmres_solver>(c);
        } else if (solver == "superlu") {
            calc1<T, superlu_solver>(c);
        } else if (solver == "jacobi") {
            calc1<T, jacobi_solver>(c);
        } else {
            calc1<T, superlu_solver>(c);
        }
    } else {
        using T = double;
        if (solver == "gmres") {
            calc1<T, gmres_solver>(c);
        } else if (solver == "superlu") {
            calc1<T, superlu_solver>(c);
        } else if (solver == "jacobi") {
            calc1<T, jacobi_solver>(c);
        } else {
            calc1<T, umfpack_solver>(c);
        }
    }

    return 0;
}
