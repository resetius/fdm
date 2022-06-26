#pragma once

#include <cmath>

#include "asp_misc.h"
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

template<typename T>
struct TSC2 {
    using matrix = T[3][3];
    static constexpr int order = 2;
    static constexpr int n = 3;

    // TSC 2d simple code
    void distribute(matrix M, T x, T y, int* x0, int* y0, T h) {
        using asp::sq;

        // nearest point
        int j = round(x / h);
        int k = round(y / h);
        x /= h;
        y /= h;

        *x0 = j-1;
        *y0 = k-1;

        // center
        double c[2] = {0.75 - sq(j - x), 0.75 - sq(k - y)};
        // right
        double r[2] = {0.5*sq(1.5 - (j+1) + x), 0.5*sq(1.5 - (k+1) + y)};
        // left
        double l[2] = {0.5*sq(1.5 - x + (j-1)), 0.5*sq(1.5 - y + (k-1))};

        M[0][0] = l[1]*l[0];
        M[0][1] = l[1]*c[0];
        M[0][2] = l[1]*r[0];

        M[1][0] = c[1]*l[0];
        M[1][1] = c[1]*c[0];
        M[1][2] = c[1]*r[0];

        M[2][0] = r[1]*l[0];
        M[2][1] = r[1]*c[0];
        M[2][2] = r[1]*r[0];
    }
};

} // namespace fdm
