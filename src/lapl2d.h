#pragma once

#include <vector>

#include "tensor.h"
#include "fft.h"
#include "blas.h"
#include "asp_misc.h"

namespace fdm {

template<typename T, bool check>
class Lapl2d {
public:
    using matrix = tensor<T,2,check>;
    const double dx, dy;
    const double dx2, dy2;

    const double lx, ly;

    const int nx, ny;

    std::vector<T> L,D,U;

    FFTTable<T> ft_table;
    FFT<T> ft;

    /**
       \param dx, dy - расстояние между точками
       \param lx, ly - расстояние между первой и последне краевой точкой по осям x,y
       \param nx, ny - 0 первая точка, nx+1 последняя (края)
     */
    Lapl2d(double dx, double dy,
           double lx, double ly,
           int nx, int ny)
        : dx(dx), dy(dy)
        , dx2(dx*dx), dy2(dy*dy)
        , lx(lx), ly(ly)
        , nx(nx), ny(ny)
        , L(nx-1), D(nx), U(nx-1)
        , ft_table(ny+1)
        , ft(ft_table, ny+1)
    { }

    /**
       \param rhs, ans - массивы размера (ny-1)*(nx-1) - только внутренние точки
     */
    void solve(T* ans, T* rhs) {
        matrix ANS({1,ny,1,nx}, ans);
        matrix RHS({1,ny,1,nx}, rhs);
        matrix RHSm({1,ny,1,nx});
        std::vector<T> s(ny+1), S(ny+1);

        for (int j = 1; j <= nx; j++) {
            for (int k = 1; k <= ny; k++) {
                s[k] = RHS[k][j];
            }

            ft.sFFT(&S[0], &s[0], dy*sqrt(2./ly));

            for (int k = 1; k <= ny; k++) {
                RHSm[k][j] = S[k];
            }
        }

        for (int k = 1; k <= ny; k++) {
            // solver
            init_Mat(k);
            int info;
            lapack::gtsv(nx, 1, &L[0], &D[0], &U[0], &RHSm[k][1], nx, &info);
            verify(info == 0);
        }

        for (int j = 1; j <= nx; j++) {
            for (int k = 1; k <= ny; k++) {
                s[k] = RHSm[k][j];

                ft.sFFT(&S[0], &s[0], sqrt(2./ly));
            }

            for (int k = 1; k <= ny; k++) {
                ANS[k][j] = S[k];
            }
        }
    }

private:
    void init_Mat(int i) {
        int di, li, ui;
        di = li = ui = 0;
        double lm = 4./dy2*asp::sq(sin(i*M_PI*0.5/(ny+1)));
        for (int j = 1; j <= nx; j++) {
            D[di++] = -2/dx2-lm;
            if (j > 1) {
                L[li++] = 1/dx2;
            }
            if (j < nx) {
                U[ui++] = 1/dx2;
            }
        }
    }
};

} // namespace fdm
