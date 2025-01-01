#include <cmath>

#include "lapl_cyl.h"
#include "cyclic_reduction.h"

using namespace asp;

namespace fdm {

template<typename T, bool check, tensor_flag zflag, bool use_cyclic_reduction>
void LaplCyl3FFT2<T,check,zflag,use_cyclic_reduction>::solve(T* ans, T* rhs) {
    RHS.use(rhs); ANS.use(ans);

#pragma omp parallel for
    for (int k = z1; k <= zn; k++) {
        for (int j = 1; j <= nr; j++) {
            auto* in = s.data();
            auto* out = S.data();
            for (int i = 0; i < nphi; i++) {
                in[i] = RHS[i][k][j];
            }

            ft_phi.pFFT_1(out, in, dphi*SQRT_M_1_PI);

            for (int i = 0; i < nphi; i++) {
                RHSm[i][k][j] = out[i];
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < nphi; i++) {
        for (int j = 1; j <= nr; j++) {
            auto* in = s.data();
            auto* out = S.data();
            for (int k = z1; k <= zn; k++) {
                in[k] = RHSm[i][k][j];
            }

            if constexpr(zflag==tensor_flag::none) {
                ft_z.sFFT(out, in, dz*slz);
            } else {
                ft_z.pFFT_1(out, in, dz*slz);
            }

            for (int k = z1; k <= zn; k++) {
                RHSm[i][k][j] = out[k];
            }
        }
    }

    // solve
#pragma omp parallel for
    for (int i = 0; i < nphi; i++) {
        for (int k = z1; k <= zn; k++) {
            if constexpr (use_cyclic_reduction) {
                int li, di, ui; li = di = ui = 0;
                T* L = &matrices[i][k][0*nr];
                T* D = &matrices[i][k][1*nr];
                T* U = &matrices[i][k][2*nr];

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

                cyclic_reduction_general(D, L, U, &RHSm[i][k][1], nrq, nr);
                // cyclic_reduction(D, L, U, &RHSm[i][k][1], q, nr);
            } else {
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
    }

#pragma omp parallel for
    for (int i = 0; i < nphi; i++) {
        for (int j = 1; j <= nr; j++) {
            auto* in = s.data();
            auto* out = S.data();

            for (int k = z1; k <= zn; k++) {
                in[k] = RHSm[i][k][j];
            }

            if constexpr(zflag==tensor_flag::none) {
                ft_z.sFFT(out, in, slz);
            } else {
                ft_z.pFFT(out, in, slz);
            }

            for (int k = z1; k <= zn; k++) {
                ANS[i][k][j] = out[k];
            }
        }
    }

#pragma omp parallel for
    for (int k = z1; k <= zn; k++) {
        for (int j = 1; j <= nr; j++) {
            auto* in = s.data();
            auto* out = S.data();

            for (int i = 0; i < nphi; i++) {
                in[i] = ANS[i][k][j];
            }

            ft_phi.pFFT(out, in, SQRT_M_1_PI);

            for (int i = 0; i < nphi; i++) {
                ANS[i][k][j] = out[i];
            }
        }
    }
}

template<typename T, bool check,tensor_flag zflag, bool use_cyclic_reduction>
void LaplCyl3FFT2<T,check,zflag,use_cyclic_reduction>::init_solver() {
    for (int i = 0; i < nphi; i++) {
        lm_phi[i] = 4.0/dphi2*sq(sin(i*dphi*0.5));
    }
    for (int k = 0; k < zpoints; k++) {
        if constexpr(zflag==tensor_flag::none) {
            lm_z[k] = 4./dz2*sq(sin(k*M_PI*0.5/zpoints));
        } else {
            lm_z[k] = 4./dz2*sq(sin(k*M_PI/zpoints));
        }
    }

    for (int i = 0; i < nphi; i++) {
        for (int k = z1; k <= zn; k++) {
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
}

template class LaplCyl3FFT2<double,true,tensor_flag::none>;
template class LaplCyl3FFT2<double,false,tensor_flag::none>;
template class LaplCyl3FFT2<float,true,tensor_flag::none>;
template class LaplCyl3FFT2<float,false,tensor_flag::none>;

template class LaplCyl3FFT2<double,true,tensor_flag::periodic>;
template class LaplCyl3FFT2<double,false,tensor_flag::periodic>;
template class LaplCyl3FFT2<float,true,tensor_flag::periodic>;
template class LaplCyl3FFT2<float,false,tensor_flag::periodic>;

} // namespace fdm
