#pragma once

#include <vector>

#include "tensor.h"
#include "fft.h"
#include "asp_misc.h"

namespace fdm {

template<typename T, bool check>
class LaplRect {
public:
    using matrix = tensor<T,2,check>;
    const double dx, dy;
    const double dx2, dy2;

    const double lx, ly;
    const double slx, sly;

    const int nx, ny;

    std::vector<T> L,D,U;

    FFTTable<T> ft_y_table;
    FFT<T> ft_y;
    FFTTable<T>  ft_x_table_;
    FFTTable<T>* ft_x_table;
    FFT<T> ft_x;

    std::vector<T> lm_y;
    std::vector<T> lm_x_;
    T* lm_x;

    /**
       \param dx, dy - расстояние между точками
       \param lx, ly - расстояние между первой и последне краевой точкой по осям x,y
       \param nx, ny - 0 первая точка, nx+1 последняя (края)
     */
    LaplRect(double dx, double dy,
             double lx, double ly,
             int nx, int ny)
        : dx(dx), dy(dy)
        , dx2(dx*dx), dy2(dy*dy)
        , lx(lx), ly(ly)
        , slx(sqrt(2./lx)), sly(sqrt(2./ly))
        , nx(nx), ny(ny)
        , L(nx-1), D(nx), U(nx-1)
        , ft_y_table(ny+1)
        , ft_y(ft_y_table, ny+1)
        , ft_x_table_(nx == ny ? 1 : ny+1)
        , ft_x_table(nx == ny ? &ft_y_table: &ft_x_table_)
        , ft_x(*ft_x_table, nx+1)
    {
        init_lm();
    }

    /**
       \param rhs, ans - массивы размера (ny-1)*(nx-1) - только внутренние точки
     */
    void solve(T* ans, T* rhs);

private:
    void init_lm();

    void init_Mat(int i);
};

template<typename T, bool check>
class LaplRectFFT2: public LaplRect<T,check> {
public:
    LaplRectFFT2(double dx, double dy,
             double lx, double ly,
             int nx, int ny)
        : LaplRect<T,check>(dx, dy, lx, ly, nx, ny)
    { }

    void solve(T* ans, T* rhs);
};

} // namespace fdm
