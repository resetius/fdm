#include <netcdf.h>
#include "ns_cyl.h"

#include "arpack_solver.h"
#include "umfpack_solver.h"
#include "gmres_solver.h"
#include "superlu_solver.h"
#include "jacobi_solver.h"

using namespace std;

template<typename T, template<typename> class Solver, bool check>
void calc(const Config& c) {
    using namespace std::chrono;

    using Task = NSCyl<T, Solver, check, tensor_flag::periodic>;
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
#define nc_call(expr) do { \
            int code = expr; \
            verify(code == 0, nc_strerror(code)); \
        } while(0);

        string filename = format(
            "eigenvectors_%f_%d_%d_%d.nc",
            ns.Re, ns.nr, ns.nz, ns.nphi);

        int ncid;
        nc_call(nc_create(filename.c_str(), NC_CLOBBER, &ncid));
        int phi_dim, z_dim, r_dim, r0_dim;

        nc_call(nc_def_dim(ncid, "phi", nphi, &phi_dim));
        nc_call(nc_def_dim(ncid, "z", nz, &z_dim));
        nc_call(nc_def_dim(ncid, "r", nr, &r_dim));
        nc_call(nc_def_dim(ncid, "r0", nr-1, &r0_dim));

        int type = NC_DOUBLE;
        if constexpr(is_same<float,T>::value) {
            type = NC_FLOAT;
        }

        vector<int> uids, vids, wids, pids;
        for (int  i = 0; i < count; i++) {
            int udims[] = {phi_dim, z_dim, r0_dim}; int uid;
            nc_call(nc_def_var(ncid, format("u_%d", i).c_str(), type, 3, udims, &uid));
            int vdims[] = {phi_dim, z_dim, r_dim}; int vid;
            nc_call(nc_def_var(ncid, format("v_%d", i).c_str(), type, 3, vdims, &vid));
            int wdims[] = {phi_dim, z_dim, r_dim}; int wid;
            nc_call(nc_def_var(ncid, format("w_%d", i).c_str(), type, 3, wdims, &wid));
            int pdims[] = {phi_dim, z_dim, r_dim}; int pid;
            nc_call(nc_def_var(ncid, format("p_%d", i).c_str(), type, 3, pdims, &pid));
            uids.push_back(uid);
            vids.push_back(vid);
            wids.push_back(wid);
            pids.push_back(pid);
        }

        nc_call(nc_enddef(ncid));

        for (int  i = 0; i < count; i++) {
            int j = indices[i];
            off = 0;
            if constexpr(is_same<float,T>::value) {
                nc_call(nc_put_var_float(ncid, uids[i], &eigenvectors[j][off]));
                off += u.size;
                nc_call(nc_put_var_float(ncid, vids[i], &eigenvectors[j][off]));
                off += v.size;
                nc_call(nc_put_var_float(ncid, wids[i], &eigenvectors[j][off]));
                off += w.size;
                nc_call(nc_put_var_float(ncid, pids[i], &eigenvectors[j][off]));
                off += p.size;
            } else {
                nc_call(nc_put_var_double(ncid, uids[i], &eigenvectors[j][off]));
                off += u.size;
                nc_call(nc_put_var_double(ncid, vids[i], &eigenvectors[j][off]));
                off += v.size;
                nc_call(nc_put_var_double(ncid, wids[i], &eigenvectors[j][off]));
                off += w.size;
                nc_call(nc_put_var_double(ncid, pids[i], &eigenvectors[j][off]));
                off += p.size;
            }
        }

        nc_call(nc_close(ncid));

#undef nc_call
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
