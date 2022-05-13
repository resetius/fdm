#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <plplot.h>

#include "matrix.h"

namespace fdm {

class matrix_plotter {
    double** data;
    double* levels;
    int nlevels;
    int rows;
    int rs;

    static void transform(double x, double y, double* tx, double* ty, void* data);

public:
    template<typename T>
    matrix_plotter(const matrix<T>& matrix) {
        data = (double**)malloc(matrix.rows*matrix.rs*sizeof(double*));
        for (int i = 0; i < matrix.rows; i++) {
            data[i] = (double*)malloc(matrix.rs*sizeof(double));
            memcpy(data[i], &matrix.vec[i*matrix.rs], matrix.rs*sizeof(double));
        }
        rows = matrix.rows;
        rs = matrix.rs;

        double mn, mx;
        mn = mx = data[0][0];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < rs; j++) {
                mn = fmin(mn, data[i][j]);
                mx = fmax(mx, data[i][j]);
            }
        }

        nlevels = 10;
        levels = (double*)malloc(nlevels*sizeof(double));
        memset(levels, 0, nlevels*sizeof(double));

        for (int i = 0; i < nlevels; i++) {
            levels[i] = mn + i * (mx-mn)/(nlevels+1);
        }
    }

    ~matrix_plotter();

    void plot();
};

} // namespace fdm
