#pragma once

#include <cmath>
#include "blas.h"

namespace fdm {

// Golub, p202, alg 5.2.5
template<typename T, typename F>
void mgsch(T& A, int n, int m, const F& dot = [](const T* x, const T* y, int n) {
    return blas::dot(n, x, 1, y, 1);
})
{
    for (int k = 0; k < n; k++) {
        auto Rkk = std::sqrt(dot(&A[k][0], &A[k][0], m));
        blas::scal(m, 1./Rkk, &A[k][0], 1);
        for (int j = k+1; j < n; j++) {
            auto Rkj = dot(&A[k][0], &A[j][0], m);
            blas::axpy(m, -Rkj, &A[k][0], 1, &A[j][0], 1);
        }
    }
}

} // namespace fdm
