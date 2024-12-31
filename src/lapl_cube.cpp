#include <cmath>
#include "lapl_cube.h"

using namespace asp;

namespace fdm {

template<typename T, bool check, typename F>
void LaplCube<T,check,F>::solve(T* ans, T* rhs) {
    tensor ANS(indices, ans);
    tensor RHS(indices, rhs);

#pragma omp parallel for
    for (int k = y1; k <= yn; k++) {
        for (int j = x1; j <= xn; j++) {
            for (int i = z1; i <= zn; i++) {
                s[k*(nz+1)+i] = RHS[i][k][j];
            }

            if constexpr(has_tensor_flag(zflag,tensor_flag::periodic)) {
                ft_z.pFFT_1(&S[k*(nz+1)], &s[k*(nz+1)], dz*slz);
            } else {
                ft_z.sFFT(&S[k*(nz+1)], &s[k*(nz+1)], dz*slz);
            }

            for (int i = z1; i <= zn; i++) {
                RHSm[i][k][j] = S[k*(nz+1)+i];
            }
        }
    }

#pragma omp parallel for
    for (int i = z1; i <= zn; i++) {
        for (int j = x1; j <= xn; j++) {
            for (int k = y1; k <= yn; k++) {
                s[i*(ny+1)+k] = RHSm[i][k][j];
            }

            if constexpr(has_tensor_flag(yflag,tensor_flag::periodic)) {
                ft_y.pFFT_1(&S[i*(ny+1)], &s[i*(ny+1)], dy*sly);
            } else {
                ft_y.sFFT(&S[i*(ny+1)], &s[i*(ny+1)], dy*sly);
            }

            for (int k = y1; k <= yn; k++) {
                RHSm[i][k][j] = S[i*(ny+1)+k];
            }
        }
    }

#pragma omp parallel for
    for (int i = z1; i <= zn; i++) {
        for (int k = y1; k <= yn; k++) {
            for (int j = x1; j <= xn; j++) {
                s[i*(nx+1)+j] = RHSm[i][k][j];
            }

            if constexpr(has_tensor_flag(xflag,tensor_flag::periodic)) {
                ft_x.pFFT_1(&S[i*(nx+1)], &s[i*(nx+1)], dx*slx);
            } else {
                ft_x.sFFT(&S[i*(nx+1)], &s[i*(nx+1)], dx*slx);
            }

            for (int j = x1; j <= xn; j++) {
                RHSm[i][k][j] = S[i*(nx+1)+j];
            }
        }
    }

#pragma omp parallel for
    for (int i = z1; i <= zn; i++) {
        for (int k = y1; k <= yn; k++) {
            for (int j = x1; j <= xn; j++) {
                T k2 = lm_z[i]+lm_y[k]+lm_x[j];
                //T rcrit = 1.5625; //0.15625;
                //RHSm[i][k][j] /= -k2 * exp (-k2*rcrit*rcrit);
                RHSm[i][k][j] /= -k2;
            }
        }
    }

    if (x1 == 0 && y1 == 0 && z1 == 0) {
        RHSm[0][0][0] = 0;
    }

#pragma omp parallel for
    for (int i = z1; i <= zn; i++) {
        for (int k = y1; k <= yn; k++) {
            for (int j = x1; j <= xn; j++) {
                s[i*(nx+1)+j] = RHSm[i][k][j];
            }

            if constexpr(has_tensor_flag(xflag,tensor_flag::periodic)) {
                ft_x.pFFT(&S[i*(nx+1)], &s[i*(nx+1)], slx);
            } else {
                ft_x.sFFT(&S[i*(nx+1)], &s[i*(nx+1)], slx);
            }

            for (int j = x1; j <= xn; j++) {
                ANS[i][k][j] = S[i*(nx+1)+j];
            }
        }
    }

#pragma omp parallel for
    for (int i = z1; i <= zn; i++) {
        for (int j = x1; j <= xn; j++) {
            for (int k = y1; k <= yn; k++) {
                s[i*(ny+1)+k] = ANS[i][k][j];
            }

            if constexpr(has_tensor_flag(yflag,tensor_flag::periodic)) {
                ft_y.pFFT(&S[i*(ny+1)], &s[i*(ny+1)], sly);
            } else {
                ft_y.sFFT(&S[i*(ny+1)], &s[i*(ny+1)], sly);
            }

            for (int k = y1; k <= yn; k++) {
                ANS[i][k][j] = S[i*(ny+1)+k];
            }
        }
    }

#pragma omp parallel for
    for (int k = y1; k <= yn; k++) {
        for (int j = x1; j <= xn; j++) {
            for (int i = z1; i <= zn; i++) {
                s[k*(nz+1)+i] = ANS[i][k][j];
            }

            if constexpr(has_tensor_flag(zflag,tensor_flag::periodic)) {
                ft_z.pFFT(&S[k*(nz+1)], &s[k*(nz+1)], slz);
            } else {
                ft_z.sFFT(&S[k*(nz+1)], &s[k*(nz+1)], slz);
            }

            for (int i = z1; i <= zn; i++) {
                ANS[i][k][j] = S[k*(nz+1)+i];
            }
        }
    }
}

template<typename T, bool check,typename F>
void LaplCube<T,check,F>::init_lm() {
    lm_y.resize(ny+1);
    for (int k = y1; k <= yn; k++) {
        if constexpr(has_tensor_flag(yflag,tensor_flag::periodic)) {
            lm_y[k] = 4./dy2*sq(sin(k*M_PI/(ny)));
        } else {
            lm_y[k] = 4./dy2*sq(sin(k*M_PI*0.5/(ny+1)));
        }
    }
    lm_x_.resize(nx+1);
    for (int j = x1; j <= xn; j++) {
        if constexpr(has_tensor_flag(xflag,tensor_flag::periodic)) {
            lm_x_[j] = 4./dx2*sq(sin(j*M_PI/(nx)));
        } else {
            lm_x_[j] = 4./dx2*sq(sin(j*M_PI*0.5/(nx+1)));
        }
    }
    lm_x = xpoints == ypoints ? &lm_y[0] : &lm_x_[0];
    lm_z_.resize(nz+1);
    for (int i = z1; i <= zn; i++) {
        if constexpr(has_tensor_flag(zflag,tensor_flag::periodic)) {
            lm_z_[i] = 4./dz2*sq(sin(i*M_PI/(nz)));
        } else {
            lm_z_[i] = 4./dz2*sq(sin(i*M_PI*0.5/(nz+1)));
        }
    }
    lm_z = zpoints == ypoints ? &lm_y[0] : &lm_z_[0];
}

template class LaplCube<double,true,tensor_flags<>>;
template class LaplCube<double,false,tensor_flags<>>;
template class LaplCube<float,true,tensor_flags<>>;
template class LaplCube<float,false,tensor_flags<>>;

template class LaplCube<double,true,tensor_flags<tensor_flag::periodic,tensor_flag::periodic,tensor_flag::periodic>>;
template class LaplCube<double,false,tensor_flags<tensor_flag::periodic,tensor_flag::periodic,tensor_flag::periodic>>;
template class LaplCube<float,true,tensor_flags<tensor_flag::periodic,tensor_flag::periodic,tensor_flag::periodic>>;
template class LaplCube<float,false,tensor_flags<tensor_flag::periodic,tensor_flag::periodic,tensor_flag::periodic>>;

} // namespace fdm
