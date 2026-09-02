#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include "blas.h"

namespace fdm {

/**
  \param A - матрица (или вектор веторов)
  \param n - число векторов
  \param m - размерность вектора
  \param dot - скалярное произведение
 */
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

/**
   Two-pass modified Gram--Schmidt with a rank check.

   \return Smallest norm after orthogonalization divided by the original
           column norm, or zero if a column is numerically dependent.
 */
template<typename F, typename T, typename Dot>
F mgsch_checked(T& A, int n, int m, F relative_tolerance, Dot dot) {
    F min_relative_norm = std::numeric_limits<F>::max();
    for (int k = 0; k < n; ++k) {
        const F original_norm = std::sqrt(dot(&A[k][0], &A[k][0], m));
        if (!(original_norm > F(0))
            || !std::isfinite(static_cast<double>(original_norm))) {
            return F(0);
        }

        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < k; ++j) {
                const F coefficient = dot(&A[j][0], &A[k][0], m);
                blas::axpy(m, -coefficient, &A[j][0], 1, &A[k][0], 1);
            }
        }

        const F norm = std::sqrt(dot(&A[k][0], &A[k][0], m));
        const F relative_norm = norm/original_norm;
        if (!(relative_norm > relative_tolerance)
            || !std::isfinite(static_cast<double>(relative_norm))) {
            return F(0);
        }
        min_relative_norm = std::min(min_relative_norm, relative_norm);
        blas::scal(m, F(1)/norm, &A[k][0], 1);
    }
    return min_relative_norm;
}

template<typename F, typename T>
F mgsch_checked(T& A, int n, int m, F relative_tolerance) {
    return mgsch_checked<F>(A, n, m, relative_tolerance,
        [](const F* x, const F* y, int size) {
            return blas::dot(size, x, 1, y, 1);
        });
}

} // namespace fdm
