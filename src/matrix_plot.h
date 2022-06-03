#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <plplot.h>

#include "matrix.h"
#include "tensor.h"
#include "verify.h"

namespace fdm {

class matrix_plotter {
    struct data {
        double** d;
        int rows;
        int rs;

        template<typename T>
        data(const matrix<T>& matrix) {
            // transposed
            rows = matrix.rows;
            rs = matrix.rs;
            d = (double**)malloc(rows*rs*sizeof(double*));
            for (int i = 0; i < rows; i++) {
                d[i] = (double*)malloc(rs*sizeof(double));
                memcpy(d[i], &matrix.vec[i*rs], rs*sizeof(double));
            }
        }

        template<typename T, bool B, typename C>
        data(const tensor<T,2,B,C>& matrix) {
            rs = matrix.sizes[0];
            rows = matrix.size / rs;
            verify(matrix.size % rs == 0);
            d = (double**)malloc(rows*rs*sizeof(double*));
            for (int i = 0; i < rows; i++) {
                d[i] = (double*)malloc(rs*sizeof(double));
                memcpy(d[i], &matrix.vec[i*rs], rs*sizeof(double));
            }
        }

        data() = default;
        ~data();

        void clear();

        data& operator=(data&& other) {
            clear();

            d = other.d;
            rows = other.rows;
            rs = other.rs;
            other.d = nullptr;
            return *this;
        }
    };

public:
    struct page {
        data u;
        data v;

        int nlevels = 10;
        double x1, x2;
        double y1, y2;
        std::string xlab = "Y";
        std::string ylab = "X";
        std::string tlab = "";

        page() = default;
        ~page() = default;

        template<typename T>
        page& scalar(const matrix<T>& matrix) {
            u = data(matrix);
            x1 = 0; x2 = u.rows-1;
            y1 = 0; y2 = u.rs-1;
            return *this;
        }

        template<typename T, bool B, typename C>
        page& scalar(const tensor<T,2,B,C>& matrix) {
            u = data(matrix);
            x1 = 0; x2 = u.rows-1;
            y1 = 0; y2 = u.rs-1;
            return *this;
        }

        template<typename T>
        page& vector(const matrix<T>& u_, const matrix<T>& v_) {
            u = data(u_);
            v = data(v_);
            x1 = 0; x2 = u.rows-1;
            y1 = 0; y2 = u.rs-1;
            return *this;
        }

        template<typename T>
        page& vector(const tensor<T,2>& u_, const tensor<T,2>& v_) {
            u = data(u_);
            v = data(v_);
            x1 = 0; x2 = u.rows-1;
            y1 = 0; y2 = u.rs-1;
            return *this;
        }

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
        mn = mx = p.u.d[0][0];
        for (int i = 0; i < p.u.rows; i++) {
            for (int j = 0; j < p.u.rs; j++) {
                mn = fmin(mn, p.u.d[i][j]);
                mx = fmax(mx, p.u.d[i][j]);
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
