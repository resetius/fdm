#pragma once

#include "tensor.h"
#include "fft.h"
#include "asp_misc.h"

namespace fdm {

template<typename T, bool check>
class LaplCube {
public:
    using tensor = fdm::tensor<T,3,check>;

    const double dx, dy, dz;
    const double dx2, dy2, dz2;

    const double lx, ly, lz;
    const double slx, sly, slz;

    const int nx, ny, nz;
    const int mxdim;

    const std::array<int,6> indices;

    FFTTable<T> ft_x_table;
    FFTTable<T> ft_y_table;
    FFTTable<T> ft_z_table;
    std::vector<FFT<T>> ft_x;
    std::vector<FFT<T>> ft_y_;
    std::vector<FFT<T>> ft_z_;

    std::vector<FFT<T>>& ft_y;
    std::vector<FFT<T>>& ft_z;

    std::vector<T> lm_y;
    std::vector<T> lm_x_;
    std::vector<T> lm_z_;
    T* lm_x;
    T* lm_z;

    tensor RHSm;
    std::vector<T> S,s;

    LaplCube(double dx, double dy, double dz,
             double lx, double ly, double lz,
             int nx, int ny, int nz)
        : dx(dx), dy(dy), dz(dz)
        , dx2(dx*dx), dy2(dy*dy), dz2(dz*dz)
        , lx(lx), ly(ly), lz(lz)
        , slx(sqrt(2./lx)), sly(sqrt(2./ly)), slz(sqrt(2./lz))
        , nx(nx), ny(ny), nz(nz)
        , mxdim(std::max({nx+1,ny+1,nz+1}))
        , indices({1,nz,1,ny,1,nx})

        , ft_x_table(nx+1)
        , ft_y_table((nx==ny&&nx==nz)?1:ny+1)
        , ft_z_table((nx==ny&&nx==nz)?1:nz+1)

        , ft_x(mxdim, {ft_x_table, nx+1})
        , ft_y_((nx==ny&&nx==nz)?
                0:mxdim,
                {ft_y_table, ny+1})
        , ft_z_((nx==ny&&nx==nz)?
                0:mxdim,
                {ft_z_table, nz+1})

        , ft_y((nx==ny&&nx==nz)?ft_x:ft_y_)
        , ft_z((nx==ny&&nx==nz)?ft_x:ft_z_)
        , RHSm(indices)
        , S(mxdim*mxdim), s(mxdim*mxdim)
    {
        init_lm();
    }

    void solve(T* ans, T* rhs);

private:
    void init_lm();
};

} // namespace fdm
