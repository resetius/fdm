#pragma once

#include <cmath>
#include <functional>
#include "blas.h"

namespace fdm {

// Golub, p202, alg 5.2.5
template<typename F, typename T>
void mgsch(T& A, int n, int m, F (*dot)(const F*, const F*, int n) = [](const auto* x, const auto* y, int n) {
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
