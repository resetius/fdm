#pragma once

#include <cmath>

#include "tensor.h"
#include "verify.h"

namespace fdm {

template<typename T, int dim>
struct CIC {
    // cloud in cell
    using tensor = fdm::tensor<T, dim, false>;
    static constexpr int order = 1;

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

} // namespace fdm
