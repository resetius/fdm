#include <algorithm>
#include <numeric>

#include "arpack_solver.h"
#include "sparse.h"
#include "asp_misc.h"
#include "config.h"

using namespace fdm;
using namespace std;
using namespace asp;

template<typename T>
void calc(Config c) {
    int n = c.get("test", "n", 100);
    int nev = c.get("test", "nev", 4);
    T tol = c.get("test", "tol", 1e-7);
    int maxit = c.get("test", "maxit", 10000);

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
        tol);

    solver.solve([&](T* y, const T* x) {
        mat.mul(y, x);
    }, eigenvalues, eigenvectors, nev);

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
