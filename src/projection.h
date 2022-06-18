#pragma once

#include "blas.h"

namespace fdm {

/**
   \param h - этот вектор проектируем вдоль, сюда же пишем результат
   \param e - базис (ортонормированный)
   \param n - число векторов базиса
   \param m - размерность одного вектора
 */
template<typename F, typename T>
void ortoproj_along(F* h, T& e, int n, int m, F (*dot)(const F*, const F*, int n) = [](const auto* x, const auto* y, int n) {
    return blas::dot(n, x, 1, y, 1);
}) {
    // proj on:
    // h1 = sum (h,ei)/(ei,ei) ei
    // proj off:
    // h1 = h - sum (h,ei)/(ei,ei) ei

    for (int i = 0; i < n; i++) {
        auto ei_ei = dot(&e[i][0], &e[i][0], m);
        auto h_ei = dot(&h[0], &e[i][0], m);
        blas::axpy(m, -h_ei/ei_ei, &e[i][0], 1, &h[0], 1);
    }
}

} // namespace fdm
