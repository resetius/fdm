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
    struct page {
        double** data;
        int rows;
        int rs;
        int nlevels = 10;
        double x1, x2;
        double y1, y2;
        std::string xlab = "Y";
        std::string ylab = "X";
        std::string tlab = "";

        template<typename T>
        page(const matrix<T>& matrix) {
            data = (double**)malloc(matrix.rows*matrix.rs*sizeof(double*));
            // transposed
            for (int i = 0; i < matrix.rows; i++) {
                data[i] = (double*)malloc(matrix.rs*sizeof(double));
                memcpy(data[i], &matrix.vec[i*matrix.rs], matrix.rs*sizeof(double));
            }
            rows = matrix.rows;
            rs = matrix.rs;

            x1 = 0; x2 = rs-1;
            y1 = 0; y2 = rs-1;
        }

        ~page();

        page& levels(int i) {
            nlevels = i;
            return *this;
        }

        page& bounds(double xx1, double yy1, double xx2, double yy2) {
            x1 = xx1; y1 = yy1;
            x2 = xx2; y2 = yy2;
            return *this;
        }

        page& labels(const std::string& x, const std::string& y, const std::string& t) {
            xlab = x; ylab = y; tlab = t;
            return *this;
        }

        page& tlabel(const std::string& t) {
            tlab = t;
            return *this;
        }
    };

    struct settings {
        std::string dname = "pngcairo";
        std::string fname_ = "1.png";
        int x = 1, y = 1;

        settings() = default;

        settings& devname(const std::string& d) {
            dname = d;
            return *this;
        }

        settings& fname(const std::string& f) {
            fname_ = f;
            return *this;
        }

        settings& sub(int x_, int y_) {
            x = x_;
            y = y_;
            return *this;
        }
    };

    matrix_plotter(const settings& s);
    ~matrix_plotter();


    void plot(const page& p) {
        clear();

        double mn, mx;
        mn = mx = p.data[0][0];
        for (int i = 0; i < p.rows; i++) {
            for (int j = 0; j < p.rs; j++) {
                mn = fmin(mn, p.data[i][j]);
                mx = fmax(mx, p.data[i][j]);
            }
        }

        levels = (double*)malloc(p.nlevels*sizeof(double));
        memset(levels, 0, p.nlevels*sizeof(double));

        for (int i = 0; i < p.nlevels; i++) {
            levels[i] = mn + i * (mx-mn)/(p.nlevels+1);
        }

        plot_internal(p);
    }

private:
    double* levels;
    const settings s;

    static void transform(double x, double y, double* tx, double* ty, void* data);

    void plot_internal(const page& s);
    void clear();
};

} // namespace fdm
