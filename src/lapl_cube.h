#pragma once

#include "tensor.h"
#include "fft.h"
#include "asp_misc.h"

namespace fdm {

template<typename T, bool check, typename F=tensor_flags<>>
class LaplCube {
public:
    using tensor = fdm::tensor<T,3,check,F>;

    static constexpr tensor_flag zflag = F::head;
    static constexpr tensor_flag yflag = F::tail::head;
    static constexpr tensor_flag xflag = F::tail::tail::head;

    const double dx, dy, dz;
    const double dx2, dy2, dz2;

    const double lx, ly, lz;
    const double slx, sly, slz;

    const int z1, y1, x1;
    const int zn, yn, xn;
    const int zpoints, ypoints, xpoints;

    const int nx, ny, nz;
    const int mxdim;

    const std::array<int,6> indices;

//#ifdef HAVE_FFTW3
//    using FFT = FFT_fftw3<T>;
//#else
    using FFT = FFT<T>;
//#endif

    FFTTable<T> ft_x_table;
    FFTTable<T> ft_y_table;
    FFTTable<T> ft_z_table;
    FFTOmpSafe<T,FFT> ft_x;
    FFTOmpSafe<T,FFT> ft_y_;
    FFTOmpSafe<T,FFT> ft_z_;

    FFTOmpSafe<T,FFT>& ft_y;
    FFTOmpSafe<T,FFT>& ft_z;

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

        , z1(has_tensor_flag(zflag,tensor_flag::periodic)?0:1)
        , y1(has_tensor_flag(yflag,tensor_flag::periodic)?0:1)
        , x1(has_tensor_flag(zflag,tensor_flag::periodic)?0:1)

        , zn(has_tensor_flag(zflag,tensor_flag::periodic)?nz-1:nz)
        , yn(has_tensor_flag(yflag,tensor_flag::periodic)?ny-1:ny)
        , xn(has_tensor_flag(zflag,tensor_flag::periodic)?nx-1:nx)

        , zpoints(has_tensor_flag(zflag,tensor_flag::periodic)?nz:nz+1)
        , ypoints(has_tensor_flag(yflag,tensor_flag::periodic)?ny:ny+1)
        , xpoints(has_tensor_flag(zflag,tensor_flag::periodic)?nx:nx+1)

        , nx(nx), ny(ny), nz(nz)
        , mxdim(std::max({nx+1,ny+1,nz+1}))
        , indices({1,nz,1,ny,1,nx})

        , ft_x_table(xpoints)
        , ft_y_table((xpoints==ypoints&&xpoints==zpoints)?1:ypoints)
        , ft_z_table((xpoints==ypoints&&xpoints==zpoints)?1:zpoints)
//#ifdef HAVE_FFTW3
//        , ft_x(xpoints)
//        , ft_y_(ypoints)
//        , ft_z_(zpoints)
//#else
        , ft_x(ft_x_table, xpoints)
        , ft_y_(ft_y_table, ypoints)
        , ft_z_(ft_z_table, zpoints)
//#endif
        , ft_y((xpoints==ypoints&&xpoints==zpoints)?ft_x:ft_y_)
        , ft_z((xpoints==ypoints&&xpoints==zpoints)?ft_x:ft_z_)
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
