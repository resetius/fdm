#include <cstring>
#include <cmath>

#include "jacobi_solver.h"

namespace fdm {

using namespace blas;

template<typename T>
void jacobi_solver<T>::solve(T* x, T* b) {
    // guess = x
    //memcpy(x, b, n*sizeof(T));
    //memset(x, 0, n*sizeof(T));
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

template<typename T>
jacobi_solver<T>& jacobi_solver<T>::operator=(csr_matrix<T>&& matrix) {
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

template class jacobi_solver<double>;
template class jacobi_solver<float>;

} // namespace fdm
