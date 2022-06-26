#pragma once

#include <cmath>

#include "tensor.h"
#include "verify.h"

namespace fdm {

template<typename T, int dim>
struct CIC {
    // cloud in cell generic n-dimentional code
    using tensor = fdm::tensor<T, dim, false>;
    static constexpr int order = 1;
    static constexpr int n = 2;

    template<int index, typename A>
    void distribute_(A acc, int* x0, const T* x, T h, T mult) {
        int k0 = floor(x[index]/h);
        T xx = (x[index] - k0*h)/h;
        x0[index] = k0;

        for (int k = 0; k < 2; k++) {
            if constexpr(index == 0) {
                acc[k]   = abs(mult*(1-k-xx));
            } else {
                distribute_<index-1>(acc[k], x0, x, h,   mult*(1-k-xx));
            }
        }
    }

    tensor distribute(int* x0, const T* x, T h) {
        tensor M({0,1,0,1});
        distribute_<dim-1>(M.acc, x0, x, h, 1.0);
        return M;
    }
};

template<typename T>
struct CIC2 {
    using matrix = T[2][2];
    static constexpr int order = 1;
    static constexpr int n = 2;

    // cloud in cell 2d simple code
    void distribute(matrix M, T x, T y, int* x0, int* y0, T h) {
        int j = floor(x / h);
        int k = floor(y / h);
        *x0 = j;
        *y0 = k;

        x = (x-j*h)/h;
        y = (y-k*h)/h;

        verify(0 <= x && x <= 1);
        verify(0 <= y && y <= 1);

        M[0][0] = (1-y)*(1-x);
        M[0][1] = (1-y)*(x);
        M[1][0] = (y)*(1-x);
        M[1][1] = (y)*(x);
    }
};

} // namespace fdm
