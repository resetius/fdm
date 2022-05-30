#pragma once

#include "sparse.h"
#include "blas.h"

namespace fdm {

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

    void solve(T* x, T* b) {
        using namespace fdm::blas;
        // guess = x
        // memcpy(x, b, n*sizeof(T));
        memset(x, 0, n*sizeof(T));
        T nr = 0;
        int it = 0;
        T tolrel = tol*nrm2(n, b, 1);

        T* x0 = x;
        T* x1 = &xn[0];

        for (it = 0; it < maxit; it++) {
            mat.mul(x1, x0);
            nr = 0;

#pragma omp parallel for reduction(+ : nr)
            for (int i = 0; i < n; i++) {
                x1[i] = (b[i] - x1[i] + diag[i] * x0[i]) / diag[i];
                nr += (x1[i] - x0[i])*(x1[i] - x0[i]);
            }

            std::swap(x1, x0);

            if (std::sqrt(nr) < tolrel) {
                break;
            }
        }

        memcpy(x, x0, n*sizeof(T));

        // printf("%d %e\n", it, nr);
    }

    jacobi_solver& operator=(csr_matrix<T>&& matrix) {
        mat = std::move(matrix);
        n = static_cast<int>(mat.Ap.size()) - 1;
        diag.resize(n); memset(&diag[0], 0, sizeof(T));
        for (int j = 0; j < n; j++) {
            for (int i0 = mat.Ap[j]; i0 < mat.Ap[j+1]; i0++) {
                if (mat.Ai[i0] == j) {
                    diag[j] = mat.Ax[i0]; break;
                }
            }
        }
        r.resize(n); xn.resize(n);
        return *this;
    }
};

} //namespace fdm
