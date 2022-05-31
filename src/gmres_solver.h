#pragma once

#include "sparse.h"

namespace fdm {

template<typename T>
class gmres_solver {
    int k;
    int maxit;
    int n = 0;
    T tol;
    csr_matrix<T> mat;

    std::vector<T> r,ax,h,q,gamma,s,c,y;

public:
    gmres_solver(int kdim=1000, int maxit=100, double tol=1e-6)
        : k(kdim)
        , maxit(maxit)
        , tol(tol)
    {
    }

    gmres_solver& operator=(csr_matrix<T>&& matrix)
    {
        mat = std::move(matrix);
        n = static_cast<int>(mat.Ap.size()) - 1;
        return *this;
    }

    void solve(T* x, T* b);

private:
    T algorithm6_9(T * x, const T* b, T eps);
};

} // namespace fdm
