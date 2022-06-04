#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <algorithm>
#include <numeric>
#include <type_traits>

#include "arpack_solver.h"
#include "asp_misc.h"
#include "sparse.h"

extern "C" {
#include <cmocka.h>
}

using namespace std;
using namespace fdm;
using namespace asp;

void test_laplace(void** ) {
    using T = double;

    int n = 100;
    int nev = 6;
    T tol = 1e-7;
    int maxit = 10000;
    int rndresid = 1;
    T fixedresid = 1;
    T from = -1.0;
    T to = 1.0;

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

    int it = 0;

    solver.solve([&](T* y, const T* x) {
        mat.mul(y, x);
        it ++;
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
        assert_float_equal(abs(ans - eigenvalues[j]), 0.0, 1e-7);
    }
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_laplace),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
