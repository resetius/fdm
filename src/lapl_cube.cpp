#include <cmath>
#include "lapl_cube.h"

namespace fdm {

template<typename T, bool check>
void LaplCube<T,check>::solve(T* ans, T* rhs) {
    tensor ANS(indices, ans);
    tensor RHS(indices, rhs);

#pragma omp parallel for
    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            for (int i = 1; i <= nz; i++) {
                s[k*(nz+1)+i] = RHS[i][k][j];
            }

            ft_z[k].sFFT(&S[k*(nz+1)], &s[k*(nz+1)], dz*slz);

            for (int i = 1; i <= nz; i++) {
                RHSm[i][k][j] = S[k*(nz+1)+i];
            }
        }
    }

#pragma omp parallel for
    for (int i = 1; i <= nz; i++) {
        for (int j = 1; j <= nx; j++) {
            for (int k = 1; k <= ny; k++) {
                s[i*(ny+1)+k] = RHSm[i][k][j];
            }

            ft_y[i].sFFT(&S[i*(ny+1)], &s[i*(ny+1)], dy*sly);

            for (int k = 1; k <= ny; k++) {
                RHSm[i][k][j] = S[i*(ny+1)+k];
            }
        }
    }

#pragma omp parallel for
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                s[i*(nx+1)+j] = RHSm[i][k][j];
            }

            ft_x[i].sFFT(&S[i*(nx+1)], &s[i*(nx+1)], dx*slx);

            for (int j = 1; j <= nx; j++) {
                RHSm[i][k][j] = S[i*(nx+1)+j];
            }
        }
    }

#pragma omp parallel for
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                RHSm[i][k][j] /= -lm_z[i]-lm_y[k]-lm_x[j];
            }
        }
    }

#pragma omp parallel for
    for (int i = 1; i <= nz; i++) {
        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                s[i*(nx+1)+j] = RHSm[i][k][j];
            }

            ft_x[i].sFFT(&S[i*(nx+1)], &s[i*(nx+1)], slx);

            for (int j = 1; j <= nx; j++) {
                ANS[i][k][j] = S[i*(nx+1)+j];
            }
        }
    }

#pragma omp parallel for
    for (int i = 1; i <= nz; i++) {
        for (int j = 1; j <= nx; j++) {
            for (int k = 1; k <= ny; k++) {
                s[i*(ny+1)+k] = ANS[i][k][j];
            }

            ft_y[i].sFFT(&S[i*(ny+1)], &s[i*(ny+1)], sly);

            for (int k = 1; k <= ny; k++) {
                ANS[i][k][j] = S[i*(ny+1)+k];
            }
        }
    }

#pragma omp parallel for
    for (int k = 1; k <= ny; k++) {
        for (int j = 1; j <= nx; j++) {
            for (int i = 1; i <= nz; i++) {
                s[k*(nz+1)+i] = ANS[i][k][j];
            }

            ft_z[k].sFFT(&S[k*(nz+1)], &s[k*(nz+1)], slz);

            for (int i = 1; i <= nz; i++) {
                ANS[i][k][j] = S[k*(nz+1)+i];
            }
        }
    }
}

template<typename T, bool check>
void LaplCube<T,check>::init_lm() {
    lm_y.resize(ny+1);
    for (int k = 1; k <= ny; k++) {
        lm_y[k] = 4./dy2*asp::sq(sin(k*M_PI*0.5/(ny+1)));
    }
    lm_x_.resize(nx+1);
    for (int j = 1; j <= ny; j++) {
        lm_x_[j] = 4./dx2*asp::sq(sin(j*M_PI*0.5/(nx+1)));
    }
    lm_x = nx == ny ? &lm_y[0] : &lm_x_[0];
    lm_z_.resize(nz+1);
    for (int i = 1; i <= nz; i++) {
        lm_z_[i] = 4./dz2*asp::sq(sin(i*M_PI*0.5/(nz+1)));
    }
    lm_z = nz == ny ? &lm_y[0] : &lm_z_[0];
}

template class LaplCube<double,true>;
template class LaplCube<double,false>;
template class LaplCube<float,true>;
template class LaplCube<float,false>;

} // namespace fdm
