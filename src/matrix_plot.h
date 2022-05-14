#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <plplot.h>

#include "matrix.h"

namespace fdm {

class matrix_plotter {
public:
    matrix_plotter();
    ~matrix_plotter();

    struct settings {
        double** data;
        int rows;
        int rs;
        int nlevels;
        std::string dname = "pngcairo";
        std::string fname_ = "1.png";
        double x1, x2;
        double y1, y2;

        template<typename T>
        settings(const matrix<T>& matrix) {
            data = (double**)malloc(matrix.rows*matrix.rs*sizeof(double*));
            for (int i = 0; i < matrix.rows; i++) {
                data[i] = (double*)malloc(matrix.rs*sizeof(double));
                memcpy(data[i], &matrix.vec[i*matrix.rs], matrix.rs*sizeof(double));
            }
            rows = matrix.rows;
            rs = matrix.rs;

            x1 = 0; x2 = rs-1;
            y1 = 0; y2 = rs-1;
        }

        ~settings();

        settings& levels(int i) {
            nlevels = i;
            return *this;
        }

        settings& devname(const std::string& d) {
            dname = d;
            return *this;
        }

        settings& fname(const std::string& f) {
            fname_ = f;
            return *this;
        }

        settings& bounds(double xx1, double yy1, double xx2, double yy2) {
            x1 = xx1; y1 = yy1;
            x2 = xx2; y2 = yy2;
            return *this;
        }
    };

    void plot(const settings& s) {
        clear();

        double mn, mx;
        mn = mx = s.data[0][0];
        for (int i = 0; i < s.rows; i++) {
            for (int j = 0; j < s.rs; j++) {
                mn = fmin(mn, s.data[i][j]);
                mx = fmax(mx, s.data[i][j]);
            }
        }

        levels = (double*)malloc(s.nlevels*sizeof(double));
        memset(levels, 0, s.nlevels*sizeof(double));

        for (int i = 0; i < s.nlevels; i++) {
            levels[i] = mn + i * (mx-mn)/(s.nlevels+1);
        }

        plot_internal(s);
    }

private:
    double* levels;

    static void transform(double x, double y, double* tx, double* ty, void* data);

    void plot_internal(const  settings& s);
    void clear();
};

} // namespace fdm
