#include <cmath>
#include <type_traits>

#include "lapl_rect.h"
#include "blas.h"

using namespace asp;
using namespace std;

namespace fdm {

template<typename T, bool check, typename F>
void LaplRect<T,check,F>::init_lm() {
    lm_y.resize(ny+1);
    for (int k = y1; k <= yn; k++) {
        if constexpr(has_tensor_flag(F::head,tensor_flag::periodic)) {
            lm_y[k] = 4./dy2*sq(sin(k*M_PI/ny));
        } else {
            lm_y[k] = 4./dy2*sq(sin(k*M_PI*0.5/(ny+1)));
        }
    }
}

template<typename T, bool check, typename F>
void LaplRectFFT2<T,check,F>::init_lm() {
    base::init_lm();
    lm_x_.resize(xpoints);
    for (int j = x1; j <= xn; j++) {
        if constexpr(has_tensor_flag(F::tail::head,tensor_flag::periodic)) {
            lm_x_[j] = 4./this->dx2*sq(sin(j*M_PI/xpoints));
        } else {
            lm_x_[j] = 4./this->dx2*sq(sin(j*M_PI*0.5/xpoints));
        }
    }
    if constexpr(is_same<F,tensor_flags<>>::value) {
        lm_x = xpoints == this->ypoints ? &this->lm_y[0] : &lm_x_[0];
    } else {
        lm_x = &lm_x_[0];
    }
}

template<typename T, bool check, typename F>
void LaplRect<T,check,F>::init_Mat(int i) {
    int di, li, ui;
    di = li = ui = 0;
    // TODO: matrix for periodic over x
    for (int j = 1; j <= nx; j++) {
        D[i*nx+di++] = -2/dx2-lm_y[i]*lm_y_scale[j];
        if (j > 1) {
            L[i*nx+li++] = L_scale[j]/dx2;
        }
        if (j < nx) {
            U[i*nx+ui++] = U_scale[j]/dx2;
        }
    }
    verify(di == nx);
    verify(ui == nx-1);
    verify(li == nx-1);
}

template<typename T, bool check, typename F>
void LaplRect<T,check,F>::solve(T* ans, T* rhs) {
    matrix ANS({y1,yn,1,nx}, ans);
    matrix RHS({y1,yn,1,nx}, rhs);
    matrix RHSm({y1,yn,1,nx});

#pragma omp parallel for
    for (int j = 1; j <= nx; j++) {
        for (int k = y1; k <= yn; k++) {
            s[j*ypoints+k] = RHS[k][j];
        }

        if constexpr(has_tensor_flag(F::head,tensor_flag::periodic)) {
            ft_y[j].pFFT_1(&S[j*ypoints], &s[j*ypoints], dy*sly);
        } else {
            ft_y[j].sFFT(&S[j*ypoints], &s[j*ypoints], dy*sly);
        }

        for (int k = y1; k <= yn; k++) {
            RHSm[k][j] = S[j*ypoints+k];
        }
    }

#pragma omp parallel for
    for (int k = y1; k <= yn; k++) {
        // solver
        init_Mat(k);
        int info;
        lapack::gtsv(nx, 1, &L[k*nx], &D[k*nx], &U[k*nx], &RHSm[k][1], nx, &info);
        verify(info == 0);
    }

#pragma omp parallel for
    for (int j = 1; j <= nx; j++) {
        for (int k = y1; k <= yn; k++) {
            s[j*ypoints+k] = RHSm[k][j];
        }

        if constexpr(has_tensor_flag(F::head,tensor_flag::periodic)) {
            ft_y[j].pFFT(&S[j*ypoints], &s[j*ypoints], sly);
        } else {
            ft_y[j].sFFT(&S[j*ypoints], &s[j*ypoints], sly);
        }

        for (int k = y1; k <= yn; k++) {
            ANS[k][j] = S[j*ypoints+k];
        }
    }
}

template<typename T, bool check, typename F>
void LaplRectFFT2<T,check,F>::solve(T* ans, T* rhs) {
    int ypoints = this->ypoints;
    int y1 = this->y1; int yn = this->yn;
    auto dy = this->dy; auto dx = this->dx;
    auto sly = this->sly; auto slx = this->slx;
    auto& ft_y = this->ft_y;
    auto& lm_y = this->lm_y;
    auto& lm_y_scale = this->lm_y_scale;
    typename LaplRect<T,check>::matrix ANS({y1,yn,x1,xn}, ans);
    typename LaplRect<T,check>::matrix RHS({y1,yn,x1,xn}, rhs);
    typename LaplRect<T,check>::matrix RHSm({y1,yn,x1,xn});
    auto& s = this->s;
    auto& S = this->S;

#pragma omp parallel for
    for (int j = x1; j <= xn; j++) {
        for (int k = y1; k <= yn; k++) {
            s[j*ypoints+k] = RHS[k][j];
        }

        if constexpr(has_tensor_flag(F::head,tensor_flag::periodic)) {
            ft_y[j].pFFT_1(&S[j*ypoints], &s[j*ypoints], dy*sly);
        } else {
            ft_y[j].sFFT(&S[j*ypoints], &s[j*ypoints], dy*sly);
        }

        for (int k = y1; k <= yn; k++) {
            RHSm[k][j] = S[j*ypoints+k];
        }
    }

#pragma omp parallel for
    for (int k = y1; k <= yn; k++) {
        for (int j = x1; j <= xn; j++) {
            s[k*xpoints+j] = RHSm[k][j];
        }

        if constexpr(has_tensor_flag(F::tail::head,tensor_flag::periodic)) {
            ft_x[k].pFFT_1(&S[k*xpoints], &s[k*xpoints], dx*slx);
        } else {
            ft_x[k].sFFT(&S[k*xpoints], &s[k*xpoints], dx*slx);
        }

        for (int j = x1; j <= xn; j++) {
            RHSm[k][j] = S[k*xpoints+j];
        }
    }


#pragma omp parallel for
    for (int k = y1; k <= yn; k++) {
        for (int j = x1; j <= xn; j++) {
            RHSm[k][j] /= -lm_y[k]*lm_y_scale[j]-lm_x[j];
        }
    }

    if (y1 == 0 && x1 == 0) {
        /*hack for double-period */
        RHSm[y1][x1] = 1;
    }

#pragma omp parallel for
    for (int k = y1; k <= yn; k++) {
        for (int j = x1; j <= xn; j++) {
            s[k*xpoints+j] = RHSm[k][j];
        }

        if constexpr(has_tensor_flag(F::tail::head,tensor_flag::periodic)) {
            ft_x[k].pFFT(&S[k*xpoints], &s[k*xpoints], slx);
        } else {
            ft_x[k].sFFT(&S[k*xpoints], &s[k*xpoints], slx);
        }

        for (int j = x1; j <= xn; j++) {
            ANS[k][j] = S[k*xpoints+j];
        }
    }

#pragma omp parallel for
    for (int j = x1; j <= xn; j++) {
        for (int k = y1; k <= yn; k++) {
            s[j*ypoints+k] = ANS[k][j];
        }

        if constexpr(has_tensor_flag(F::head,tensor_flag::periodic)) {
            ft_y[j].pFFT(&S[j*ypoints], &s[j*ypoints], sly);
        } else {
            ft_y[j].sFFT(&S[j*ypoints], &s[j*ypoints], sly);
        }

        for (int k = y1; k <= yn; k++) {
            ANS[k][j] = S[j*ypoints+k];
        }
    }
}

template class LaplRectFFT2<double,true,tensor_flags<>>;
template class LaplRectFFT2<double,false,tensor_flags<>>;
template class LaplRectFFT2<float,true,tensor_flags<>>;
template class LaplRectFFT2<float,false,tensor_flags<>>;

template class LaplRectFFT2<double,true,tensor_flags<tensor_flag::periodic>>;
template class LaplRectFFT2<double,false,tensor_flags<tensor_flag::periodic>>;
template class LaplRectFFT2<float,true,tensor_flags<tensor_flag::periodic>>;
template class LaplRectFFT2<float,false,tensor_flags<tensor_flag::periodic>>;

template class LaplRectFFT2<double,true,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;
template class LaplRectFFT2<double,false,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;
template class LaplRectFFT2<float,true,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;
template class LaplRectFFT2<float,false,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;

template class LaplRect<double,true,tensor_flags<>>;
template class LaplRect<double,false,tensor_flags<>>;
template class LaplRect<float,true,tensor_flags<>>;
template class LaplRect<float,false,tensor_flags<>>;

template class LaplRect<double,true,tensor_flags<tensor_flag::periodic>>;
template class LaplRect<double,false,tensor_flags<tensor_flag::periodic>>;
template class LaplRect<float,true,tensor_flags<tensor_flag::periodic>>;
template class LaplRect<float,false,tensor_flags<tensor_flag::periodic>>;

// don't use!
template class LaplRect<double,true,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;
template class LaplRect<double,false,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;
template class LaplRect<float,true,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;
template class LaplRect<float,false,tensor_flags<tensor_flag::periodic,tensor_flag::periodic>>;

} // namespace fdm
