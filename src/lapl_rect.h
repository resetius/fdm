#pragma once

#include <vector>

#include "tensor.h"
#include "fft.h"
#include "asp_misc.h"

namespace fdm {

template<typename T, bool check, typename F = tensor_flags<>>
class LaplRect {
public:
    using matrix = tensor<T,2,check,F>;
    const double dx, dy;
    const double dx2, dy2;

    const double lx, ly;
    const double slx, sly;

    const int nx, ny;

    const int y1, yn, ypoints;

    std::vector<T> L,D,U;

    FFTTable<T> ft_y_table;
    FFTOmpSafe<T,FFT<T>> ft_y;

    std::vector<T> lm_y;

    std::vector<T> lm_y_scale, L_scale, U_scale;
    std::vector<T> S,s;

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

        , y1(has_tensor_flag(F::head,tensor_flag::periodic)?0:1)
        , yn(has_tensor_flag(F::head,tensor_flag::periodic)?ny-1:ny)
        , ypoints(has_tensor_flag(F::head,tensor_flag::periodic)?ny:ny+1)

        , L(ypoints*nx), D(ypoints*nx), U(ypoints*nx)
        , ft_y_table(ypoints)
        , ft_y(ft_y_table, ypoints)

        , lm_y_scale(nx+1, 1) // use 1/r/r for cylindrical coordinates
        , L_scale(nx+1, 1) // use (r-0.5*dr)/r for cyl...
        , U_scale(nx+1, 1) // use (r+0.5*dr)/r for cyl...
        , S((nx+1)*(ny+1)), s((nx+1)*(ny+1))
    {
        init_lm();
    }

    /**
       \param rhs, ans - массивы размера (ny-1)*(nx-1) - только внутренние точки
     */
    void solve(T* ans, T* rhs);

protected:
    void init_lm();

    void init_Mat(int i);
};

template<typename T, bool check, typename F=tensor_flags<>>
class LaplRectFFT2: public LaplRect<T,check,F> {
public:
    using base = LaplRect<T,check,F>;
    const int x1, xn, xpoints;

    FFTTable<T>  ft_x_table_;
    FFTTable<T>* ft_x_table;
    FFTOmpSafe<T,FFT<T>> ft_x;

    std::vector<T> lm_x_;
    T* lm_x;

    LaplRectFFT2(double dx, double dy,
                 double lx, double ly,
                 int nx, int ny)
        : LaplRect<T,check,F>(dx, dy, lx, ly, nx, ny)

        , x1(has_tensor_flag(F::tail::head,tensor_flag::periodic)?0:1)
        , xn(has_tensor_flag(F::tail::head,tensor_flag::periodic)?nx-1:nx)
        , xpoints(has_tensor_flag(F::tail::head,tensor_flag::periodic)?nx:nx+1)

        , ft_x_table_(xpoints == this->ypoints ? 1 : xpoints)
        , ft_x_table(xpoints == this->ypoints ? &this->ft_y_table: &ft_x_table_)
        , ft_x(*ft_x_table, xpoints)
    {
        init_lm();
    }

    void solve(T* ans, T* rhs);

private:
    void init_lm();
};

} // namespace fdm
