#include <cmath>

#include "lapl_rect.h"
#include "blas.h"

using namespace asp;

namespace fdm {

template<typename T, bool check>
void LaplRect<T,check>::init_lm() {
    lm_y.resize(ny+1);
    for (int k = 1; k <= ny; k++) {
        lm_y[k] = 4./dy2*sq(sin(k*M_PI*0.5/(ny+1)));
    }
    lm_x_.resize(nx+1);
    for (int j = 1; j <= ny; j++) {
        lm_x_[j] = 4./dx2*sq(sin(j*M_PI*0.5/(nx+1)));
    }
    lm_x = nx == ny ? &lm_y[0] : &lm_x_[0];
}

template<typename T, bool check>
void LaplRect<T,check>::init_Mat(int i) {
    int di, li, ui;
    di = li = ui = 0;
    double lm = lm_y[i];
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

template<typename T, bool check>
void LaplRect<T,check>::solve(T* ans, T* rhs) {
    matrix ANS({1,ny,1,nx}, ans);
    matrix RHS({1,ny,1,nx}, rhs);
    matrix RHSm({1,ny,1,nx});
    std::vector<T> s(ny+1), S(ny+1);

    for (int j = 1; j <= nx; j++) {
        for (int k = 1; k <= ny; k++) {
            s[k] = RHS[k][j];
        }

        ft_y.sFFT(&S[0], &s[0], dy*sqrt(2./ly));

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
        }

        ft_y.sFFT(&S[0], &s[0], sqrt(2./ly));

        for (int k = 1; k <= ny; k++) {
            ANS[k][j] = S[k];
        }
    }
}

template<typename T, bool check>
void LaplRectFFT2<T,check>::solve(T* ans, T* rhs) {
    int nx = this->nx; int ny = this->ny;
    auto dy = this->dy; auto dx = this->dx;
    auto sly = this->sly; auto slx = this->slx;
    auto& ft_y = this->ft_y; auto& ft_x = this->ft_x;
    auto& lm_y = this->lm_y; auto& lm_x = this->lm_x;
    typename LaplRect<T,check>::matrix ANS({1,ny,1,nx}, ans);
    typename LaplRect<T,check>::matrix RHS({1,ny,1,nx}, rhs);
    typename LaplRect<T,check>::matrix RHSm({1,ny,1,nx});
    std::vector<T> s(ny+1), S(ny+1);

    for (int j = 1; j <= nx; j++) {
        for (int k = 1; k <= ny; k++) {
            s[k] = RHS[k][j];
        }

        ft_y.sFFT(&S[0], &s[0], dy*sly);

        for (int k = 1; k <= ny; k++) {
            RHSm[k][j] = S[k];
        }
    }

    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            s[j] = RHSm[k][j];
        }

        ft_x.sFFT(&S[0], &s[0], dx*slx);

        for (int j = 1; j <= nx; j++) {
            RHSm[k][j] = S[j];
        }
    }

    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            RHSm[k][j] /= -lm_y[k]-lm_x[j];
        }
    }

    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            s[j] = RHSm[k][j];
        }

        ft_x.sFFT(&S[0], &s[0], slx);

        for (int j = 1; j <= nx; j++) {
            ANS[k][j] = S[j];
        }
    }

    for (int j = 1; j <= nx; j++) {
        for (int k = 1; k <= ny; k++) {
            s[k] = ANS[k][j];
        }

        ft_y.sFFT(&S[0], &s[0], sly);

        for (int k = 1; k <= ny; k++) {
            ANS[k][j] = S[k];
        }
    }
}

template class LaplRectFFT2<double,true>;
template class LaplRectFFT2<double,false>;
template class LaplRectFFT2<float,true>;
template class LaplRectFFT2<float,false>;

template class LaplRect<double,true>;
template class LaplRect<double,false>;
template class LaplRect<float,true>;
template class LaplRect<float,false>;

} // namespace fdm
