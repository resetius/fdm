#pragma once

#include <cmath>
#include <string>
#include "tensor.h"
#include "lapl_rect.h"

namespace fdm {

template<tensor_flag zflag, tensor_flag yflag = tensor_flag::none>
struct lapl_flags {
    using value = tensor_flags<zflag,yflag>;
};

template<>
struct lapl_flags<tensor_flag::periodic, tensor_flag::none> {
    using value = tensor_flags<tensor_flag::periodic>;
};

template<>
struct lapl_flags<tensor_flag::none, tensor_flag::none> {
    using value = tensor_flags<>;
};

template<typename T, bool check, typename F = tensor_flags<>>
class velocity_plotter {
    // z,y,x
    // phi,z,r
    static constexpr tensor_flag zflag = F::head;
    static constexpr tensor_flag yflag = F::tail::head;

    using matrix_x = tensor<T,2,check,typename lapl_flags<zflag,yflag>::value>;
    using matrix_y = tensor<T,2,check,typename lapl_flags<zflag>::value>;
    using matrix_z = tensor<T,2,check,typename lapl_flags<yflag>::value>;

    const int nx,ny,nz;
    const bool cyl;
    const double dx,dy,dz;

    const double ly,lz;
    const double xx1,yy1,zz1;
    const double xx2,yy2,zz2;

    const int y_,y0,y1,yn,ynn;
    const int z_,z0,z1,zn,znn;

    matrix_x RHS_x;
    matrix_y RHS_y;
    matrix_z RHS_z;

    matrix_x psi_x;
    matrix_y psi_y;
    matrix_z psi_z;

    matrix_x vx, wx;
    matrix_y uy, wy;
    matrix_z uz, vz;

    tensor<T,3,check,F> u,v,w;

    std::string Xlabel = "X";
    std::string Ylabel = "Y";
    std::string Zlabel = "Z";

    // z,y
    LaplRectFFT2<T,check,typename lapl_flags<zflag,yflag>::value> lapl_x; // lapl_r in cyl coords
    // z,x
    LaplRect<T,check,typename lapl_flags<zflag>::value> lapl_y; // lapl_z in cyl coords
    // y,x
    LaplRect<T,check,typename lapl_flags<yflag>::value> lapl_z; // lapl_phi in cyl coords

public:
    // TODO: lx,ly,lz -- вычисляем, остальное как есть
    // TODO: этот plotter работает только для сдвинутых сеток
    velocity_plotter(double dx, double dy, double dz,
                     double xx1, double yy1, double zz1,
                     double xx2, double yy2, double zz2,
                     int nx, int ny, int nz, bool cyl = false)

        : nx(nx), ny(ny), nz(nz)
        , cyl(cyl)
        , dx(dx), dy(dy), dz(dz)

        , ly(has_tensor_flag(yflag,tensor_flag::periodic)?yy2-yy1:yy2-yy1+dy)
        , lz(has_tensor_flag(zflag,tensor_flag::periodic)?zz2-zz1:zz2-zz1+dz)

        , xx1(xx1), yy1(yy1), zz1(zz1)
        , xx2(xx2), yy2(yy2), zz2(zz2)

        , y_(has_tensor_flag(yflag,tensor_flag::periodic)?0:-1)
        , y0(0)
        , y1(has_tensor_flag(yflag,tensor_flag::periodic)?0:1)
        , yn(has_tensor_flag(yflag,tensor_flag::periodic)?ny-1:ny)
        , ynn(has_tensor_flag(yflag,tensor_flag::periodic)?ny-1:ny+1)

        , z_(has_tensor_flag(zflag,tensor_flag::periodic)?0:-1)
        , z0(0)
        , z1(has_tensor_flag(zflag,tensor_flag::periodic)?0:1)
        , zn(has_tensor_flag(zflag,tensor_flag::periodic)?nz-1:nz)
        , znn(has_tensor_flag(zflag,tensor_flag::periodic)?nz-1:nz+1)

        , RHS_x({z1, zn, y1, yn})
        , RHS_y({z1, zn, 1, nx})
        , RHS_z({y1, yn, 1, nx})

        , psi_x({z1, zn, y1, yn})
        , psi_y({z1, zn, 1, nx})
        , psi_z({y1, yn, 1, nx})

        , vx({z0, znn, y0, ynn})
        , wx({z0, znn, y0, ynn})

        , uy({z0, znn, 0, nx+1})
        , wy({z0, znn, 0, nx+1})

        , uz({y0, ynn, 0, nx+1})
        , vz({y0, ynn, 0, nx+1})

        , u({z0,znn,y0,ynn,-1,nx+1}, reinterpret_cast<T*>(0xF))
        , v({z0,znn,y_,ynn, 0,nx+1}, reinterpret_cast<T*>(0xF))
        , w({z_,znn,y0,ynn, 0,nx+1}, reinterpret_cast<T*>(0xF))

        , lapl_x(dy,dz,ly,lz,ny,nz)
        , lapl_y(dx,dz,xx2-xx1+dx,lz,nx,nz)
        , lapl_z(dx,dy,xx2-xx1+dx,ly,nx,ny)

    {
        if (cyl) {
            for (int j = 1; j <= nx; j++) {
                double r = xx1+j*dx-dx/2;
                lapl_y.lm_y_scale[j] = 1./r/r;
                lapl_y.U_scale[j] = (r+dx/2)/r;
                lapl_y.L_scale[j] = (r-dx/2)/r;
            }

            for (int j = 1; j <= nx; j++) {
                double r = xx1+j*dx-dx/2;
                lapl_z.U_scale[j] = (r+dx/2)/r;
                lapl_z.L_scale[j] = (r-dx/2)/r;
            }
        }
    }

    void set_labels(const std::string& X, const std::string& Y, const std::string& Z) {
        Xlabel = X; Ylabel = Y; Zlabel = Z;
    }

    void update_slices(T* u, T* v, T* w);
    void plot(const std::string& name, double t);
    void vtk_out(const std::string& name, int step);
};

} // namespace fdm
