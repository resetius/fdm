#include <cmath>

#include "lapl_cyl.h"

using namespace asp;

namespace fdm {

template<typename T, bool check>
void LaplCyl3FFT2<T,check>::solve(T* ans, T* rhs) {
    RHS.use(rhs); ANS.use(ans);

#pragma omp parallel for
    for (int k = 1; k <= nz; k++) {
        for (int j = 1; j <= nr; j++) {
            for (int i = 0; i < nphi; i++) {
                s[k*nphi+i] = RHS[i][k][j];
            }

            ft_phi[k].pFFT_1(&S[k*nphi], &s[k*nphi], dphi*SQRT_M_1_PI);

            for (int i = 0; i < nphi; i++) {
                RHSm[i][k][j] = S[k*nphi+i];
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < nphi; i++) {
        for (int j = 1; j <= nr; j++) {
            for (int k = 1; k <= nz; k++) {
                s[i*(nz+1)+k] = RHSm[i][k][j];
            }

            ft_z[i].sFFT(&S[i*(nz+1)], &s[i*(nz+1)], dz*slz);

            for (int k = 1; k <= nz; k++) {
                RHSm[i][k][j] = S[i*(nz+1)+k];
            }
        }
    }

    // solve
#pragma omp parallel for
    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            T* L = &matrices[i][k][0*nr];
            T* D = &matrices[i][k][1*nr];
            T* U = &matrices[i][k][2*nr];
            T* U2 = &matrices[i][k][3*nr];
            int* ipiv = &ipivs[i][k][0];

            int info;
            lapack::gttrs("N", nr, 1, L, D, U, U2, ipiv, &RHSm[i][k][1], nr, &info);
            verify(info == 0);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < nphi; i++) {
        for (int j = 1; j <= nr; j++) {
            for (int k = 1; k <= nz; k++) {
                s[i*(nz+1)+k] = RHSm[i][k][j];
            }

            ft_z[i].sFFT(&S[i*(nz+1)], &s[i*(nz+1)], slz);

            for (int k = 1; k <= nz; k++) {
                ANS[i][k][j] = S[i*(nz+1)+k];
            }
        }
    }

#pragma omp parallel for
    for (int k = 1; k <= nz; k++) {
        for (int j = 1; j <= nr; j++) {
            for (int i = 0; i < nphi; i++) {
                s[k*nphi+i] = ANS[i][k][j];
            }

            ft_phi[k].pFFT(&S[k*nphi], &s[k*nphi], SQRT_M_1_PI);

            for (int i = 0; i < nphi; i++) {
                ANS[i][k][j] = S[k*nphi+i];
            }
        }
    }
}

template<typename T, bool check>
void LaplCyl3FFT2<T,check>::init_solver() {
    for (int i = 0; i < nphi; i++) {
        lm_phi[i] = 4.0/dphi2*sq(sin(i*dphi*0.5));
    }
    for (int k = 0; k <= nz; k++) {
        lm_z[k] = 4./dz2*sq(sin(k*M_PI*0.5/(nz+1)));
    }

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            int li, di, ui; li = di = ui = 0;
            T* L = &matrices[i][k][0*nr];
            T* D = &matrices[i][k][1*nr];
            T* U = &matrices[i][k][2*nr];
            T* U2 = &matrices[i][k][3*nr];
            int* ipiv = &ipivs[i][k][0];
            for (int j = 1; j <= nr; j++) {
                double r = r0+j*dr;
                D[di++] = -2/dr2-lm_phi[i]/r/r-lm_z[k];
                if (j > 1) {
                    L[li++] = (r-0.5*dr)/dr2/r;
                }
                if (j < nr) {
                    U[ui++] = (r+0.5*dr)/dr2/r;
                }
            }
            verify(di == nr);
            verify(ui == nr-1);
            verify(li == nr-1);

            int info = 0;
            lapack::gttrf(nr, L, D, U, U2, ipiv, &info);
            verify(info == 0);
        }
    }

    ft_phi.reserve(nz+1);
    for (int k = 0; k <= nz; k++) {
        ft_phi.emplace_back(ft_phi_table, nphi);
    }

    if (nphi == nz+1) {
        return;
    }

    ft_z.reserve(nphi);
    for (int k = 0; k < nphi; k++) {
        ft_z.emplace_back(*ft_z_table, nz+1);
    }
}

template class LaplCyl3FFT2<double,true>;
template class LaplCyl3FFT2<double,false>;
template class LaplCyl3FFT2<float,true>;
template class LaplCyl3FFT2<float,false>;

} // namespace fdm
