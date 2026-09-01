#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <algorithm>
#include <numeric>
#include <thread>
#include <type_traits>
#include <vector>

#include "arpack_solver.h"
#include "asp_misc.h"
#include "sparse.h"

extern "C" {
#include <cmocka.h>
}

using namespace std;
using namespace fdm;
using namespace asp;

template<typename T>
void check_laplace(T tol, T eigenvalue_tolerance) {
    int n = 100;
    int nev = 6;
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

    solver.solve([&](T* y, const T* x) {
        mat.mul(y, x);
    }, eigenvalues, eigenvectors, nev);

    assert_int_equal(solver.last_naupd_info(), 0);
    assert_int_equal(solver.last_neupd_info(), 0);
    assert_true(solver.last_nconv() >= nev);
    assert_true(solver.last_iterations() > 0);

    int nconv = static_cast<int>(eigenvalues.size());
    vector<int> indices(nconv);
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&](int a, int b) {
        return abs(eigenvalues[a]) > abs(eigenvalues[b]);
    });

    for (int  i = 0; i < nconv; i++) {
        int j = indices[i];
        complex<T> ans = complex<T>(4*sq(sin(M_PI*(n-i)/(n+1)/2.)), 0);
        assert_float_equal(abs(ans - eigenvalues[j]), 0.0,
                           eigenvalue_tolerance);
    }
}

void test_laplace_double(void**) {
    check_laplace<double>(1e-7, 1e-7);
}

void test_laplace_float(void**) {
    check_laplace<float>(1e-5f, 2e-4f);
}

// Each reverse-communication loop remains on one thread; TLS isolates loops.
template<typename T>
static vector<complex<T>> solve_shifted(T shift, T tolerance) {
    int n = 200;
    int nev = 6;

    csr_matrix<T> mat;
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            mat.add(i, i-1, -1);
        }
        mat.add(i, i, 2 + shift);
        if (i < n-1) {
            mat.add(i, i+1, -1);
        }
    }
    mat.close();

    vector<complex<T>> eigenvalues;
    vector<vector<T>> eigenvectors;

    arpack_solver<T> solver(
        n, 10000,
        arpack_solver<T>::standard,
        arpack_solver<T>::largest_magnitude,
        arpack_solver<T>::fixed,
        tolerance);
    solver.set_resid(T(1));
    solver.solve([&](T* y, const T* x) {
        mat.mul(y, x);
    }, eigenvalues, eigenvectors, nev);

    sort(eigenvalues.begin(), eigenvalues.end(), [](auto a, auto b) {
        return abs(a) > abs(b);
    });
    return eigenvalues;
}

template<typename T>
void check_concurrent_solvers(T tolerance, double comparison_tolerance) {
    const int nthreads = 8;

    vector<vector<complex<T>>> expected(nthreads);
    for (int i = 0; i < nthreads; i++) {
        expected[i] = solve_shifted(T(0.01)*T(i), tolerance);
    }

    vector<vector<complex<T>>> got(nthreads);
    vector<thread> threads;
    for (int i = 0; i < nthreads; i++) {
        threads.emplace_back([&got, i, tolerance]() {
            got[i] = solve_shifted(T(0.01)*T(i), tolerance);
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    for (int i = 0; i < nthreads; i++) {
        assert_int_equal(got[i].size(), expected[i].size());
        for (size_t j = 0; j < expected[i].size(); j++) {
            assert_float_equal(abs(expected[i][j] - got[i][j]), 0.0,
                               comparison_tolerance);
        }
    }
}

void test_concurrent_solvers_double(void**) {
    check_concurrent_solvers<double>(1e-7, 1e-12);
}

void test_concurrent_solvers_float(void**) {
    check_concurrent_solvers<float>(1e-5f, 1e-5);
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_laplace_double),
        cmocka_unit_test(test_laplace_float),
        cmocka_unit_test(test_concurrent_solvers_double),
        cmocka_unit_test(test_concurrent_solvers_float),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
