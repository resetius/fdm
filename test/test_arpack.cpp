#include "arpack_solver.h"
#include "sparse.h"
#include "asp_misc.h"

using namespace fdm;
using namespace std;
using namespace asp;

template<typename T>
void calc(int n) {
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

    vector<complex<T>> eigenvalues;
    vector<vector<T>> eigenvectors;

    arpack_solver<T> solver(
        n, 10000,
        arpack_solver<T>::standard,
        arpack_solver<T>::largest_magnitude,
        1e-7);

    int nev = 4;

    solver.solve([&](T* y, const T* x) {
        mat.mul(y, x);
    }, eigenvalues, eigenvectors, nev);

    for (int i = 0; i < static_cast<int>(eigenvalues.size()); i++) {
        printf(" -> %e %e | %e\n",
               eigenvalues[i].real(), eigenvalues[i].imag(),
               4*sq(sin(M_PI*(n-i-1)/(n+1)/2.)));
    }
}

int main() {
    int n = 100;
    calc<double>(n);
    //calc<float>(n);
    return 0;
}
