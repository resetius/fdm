#pragma once

#include "sparse.h"
#include "blas.h"

namespace fdm {

// T=float or double
template<typename T>
class jacobi_solver {
    T tol;
    int maxit;
    csr_matrix<T> mat;
    std::vector<T> diag;
    std::vector<T> xn;
    std::vector<T> r;
    int n;

public:
    jacobi_solver(double tol=1e-6, int maxit=10000)
        : tol(tol)
        , maxit(maxit)
    {
    }

    void solve(T* x, T* b);

    jacobi_solver& operator=(csr_matrix<T>&& matrix);
};

} //namespace fdm
