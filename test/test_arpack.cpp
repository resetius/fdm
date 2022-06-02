#include <algorithm>
#include <numeric>
#include <chrono>


#include "arpack_solver.h"
#include "sparse.h"
#include "asp_misc.h"
#include "config.h"

using namespace fdm;
using namespace std;
using namespace std::chrono;
using namespace asp;

template<typename T>
void calc(Config c) {
    int n = c.get("test", "n", 100);
    int nev = c.get("test", "nev", 4);
    T tol = c.get("test", "tol", 1e-7);
    int maxit = c.get("test", "maxit", 10000);
    int rndresid = c.get("resid", "rnd", 1);
    T fixedresid = c.get("resid", "fixed", 1);
    T from = c.get("resid", "a", -1.0);
    T to = c.get("resid", "b", 1.0);

    csr_matrix<T> mat;

    // laplace matrix
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            mat.add(i, i-1, -1);
        }
        mat.add(i, i, 2);
        if (i < n-1) {
            mat.add(i, i+1, -1);
        }
    }
    mat.close();

    vector<complex<T>> eigenvalues;
    vector<vector<T>> eigenvectors;

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

    solver.solve([&](T* y, const T* x) {
        mat.mul(y, x);
        it ++;
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
        complex<T> ans = complex<T>(4*sq(sin(M_PI*(n-i)/(n+1)/2.)), 0);
        printf(" -> %e %e | %e, %e\n",
               eigenvalues[j].real(), eigenvalues[j].imag(),
               ans.real(),
               abs(ans - eigenvalues[j]));
    }
}

int main(int argc, char** argv) {
    string config_fn = "test_arpack.ini";
    Config c;
    c.open(config_fn);
    c.rewrite(argc, argv);

    calc<double>(c);
    //calc<float>(c);
    return 0;
}
