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
struct CIC3 {
    using matrix = T[2][2][2];
    static constexpr int order = 1;
    static constexpr int n = 2;

    // cloud in cell
    void distribute(matrix M, T x, T y, T z, int* x0, int* y0, int* z0, T h) {
        int j = floor(x / h);
        int k = floor(y / h);
        int i = floor(z / h);
        *x0 = j;
        *y0 = k;
        *z0 = i;

        x = (x-j*h)/h;
        y = (y-k*h)/h;
        z = (z-i*h)/h;

        verify(0 <= x && x <= 1);
        verify(0 <= y && y <= 1);
        verify(0 <= z && z <= 1);

        M[0][0][0] = (1-z)*(1-y)*(1-x);
        M[0][0][1] = (1-z)*(1-y)*(x);
        M[0][1][0] = (1-z)*(y)*(1-x);
        M[0][1][1] = (1-z)*(y)*(x);

        M[1][0][0] = (z)*(1-y)*(1-x);
        M[1][0][1] = (z)*(1-y)*(x);
        M[1][1][0] = (z)*(y)*(1-x);
        M[1][1][1] = (z)*(y)*(x);
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

template<typename T>
struct PCS2 {
    // splines
    // Kun Xu, Yipeg Jing, An Accurate P3M Algorithm for Gravitational Lensing
    // Studies in Simulations, 2021
    // W(x) = 1/6 (4 - 6x^2 + 3 |x|^3), 0 <= |x| < 1
    //      = 1/6 (2 - |x|)^3, 1 <= |x| < 2
    //      = 0, otherwise
    using matrix = T[4][4];
    static constexpr int order = 3;
    static constexpr int n = 4;

    void distribute(matrix M, T x, T y, int* x0, int* y0, T h) {
        using asp::sq;
        using asp::tr;

        int j = ceil(x / h);
        int k = ceil(y / h);

        x /= h; y /= h;

        *x0 = j-2; *y0 = k-2;

        double r[2] = {
            1./6.*(4.-6.*sq(j-x)+3.*tr(j-x)),
            1./6.*(4.-6.*sq(k-y)+3.*tr(k-y))
        };
        double l[2] = {
            1./6.*(4.-6.*sq(j-1-x)+3.*tr(x-(j-1))),
            1./6.*(4.-6.*sq(k-1-y)+3.*tr(y-(k-1)))
        };
        double rr[2] = {
            1./6.*tr(2-(j+1-x)),
            1./6.*tr(2-(k+1-y))
        };
        double ll[2] = {
            1./6.*tr(2-(x-(j-2))),
            1./6.*tr(2-(y-(k-2)))
        };

        M[0][0] = ll[1]*ll[0];
        M[0][1] = ll[1]*l [0];
        M[0][2] = ll[1]*r [0];
        M[0][3] = ll[1]*rr[0];

        M[1][0] = l [1]*ll[0];
        M[1][1] = l [1]*l [0];
        M[1][2] = l [1]*r [0];
        M[1][3] = l [1]*rr[0];

        M[2][0] = r [1]*ll[0];
        M[2][1] = r [1]*l [0];
        M[2][2] = r [1]*r [0];
        M[2][3] = r [1]*rr[0];

        M[3][0] = rr[1]*ll[0];
        M[3][1] = rr[1]*l [0];
        M[3][2] = rr[1]*r [0];
        M[3][3] = rr[1]*rr[0];
    }
};

} // namespace fdm
