#pragma once

#include "tensor.h"
#include "asp_fft.h"
#include "verify.h"

namespace fdm {

template<typename T, template<typename> class Solver, bool check>
class LaplCyl3 {
    const double R, r0;
    const double h1, h2;

    const int nr, nz, nphi;
    const double dr, dz, dphi;
    const double dr2, dz2, dphi2;

    // strange
    std::vector<Solver<T>> solver;

public:
    LaplCyl3(double R, double r0, double h1, double h2,
             int nr, int nz, int nphi)
        : R(R), r0(r0)
        , h1(h1), h2(h2)
        , nr(nr), nz(nz), nphi(nphi)
        , dr((R-r0)/nr), dz((h2-h1)/nz), dphi(2*M_PI/nphi)
        , dr2(dr*dr), dz2(dz*dz), dphi2(dphi*dphi)
        , solver(nphi)
    {
        init_solver();
    }

    void solve(double* ans, const double* rhs) {
        using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

        std::vector<int> indices = {0, nphi-1, 1, nz, 1, nr};
        tensor<T,3,check,tensor_flags> RHS(indices, rhs);
        tensor<T,3,check,tensor_flags> ANS(indices, ans);
        tensor<T,3,check,tensor_flags> RHSm(indices);
        tensor<T,2,check> X({1,nz,1,nr});

        std::vector<T> s(nphi), S(nphi);

        fft* ft = FFT_init(FFT_PERIODIC, nphi);
        verify(ft);

        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                for (int i = 0; i < nphi; i++) {
                    s[i] = RHS[i][k][j];
                }

                pFFT_2(&S[0], &s[0], dphi /* check */, ft);

                for (int i = 0; i < nphi; i++) {
                    RHSm[i][k][j] = S[i];
                }
            }
        }

        for (int i = 0; i < nphi; i++) {
            // решаем i систем
            // TODO: init solver
            solver[i].solve(&X[1][1], &RHSm[i][1][1]);

            RHSm[i].assign(X.acc);
        }

        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                for (int i = 0; i < nphi; i++) {
                    s[i] = RHSm[i][k][j];
                }

                pFFT_2_1(&S[0], &s[0], dphi /* check*/, ft);

                for (int i = 0; i < nphi; i++) {
                    ANS[i][k][j] = S[i];
                }
            }
        }

        FFT_free(ft);
    }

private:
    void init_solver() {
        for (int i = 0; i < nphi; i++) {
            init_solver(i);
        }
    }

    void init_solver(int i) {
        tensor<T,2,check> RHS_phi({1,nz,1,nr});
        csr_matrix<T> P_phi;

        T lm = 1.0; // TODO: eigenvalue

        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                int id = RHS_phi.index({k,j});
                double r = r0+j*dr-dr/2;

                if (k > 1) {
                    P_phi.add(id, RHS_phi.index({k-1,j}), 1/dz2);
                }

                if (j > 1) {
                    P_phi.add(id, RHS_phi.index({k,j-1}), (r-0.5*dr)/dr2/r);
                }

                P_phi.add(id, RHS_phi.index({k,j}), -2/dr2-2/dz2+lm/r/r);

                if (j < nr) {
                    P_phi.add(id, RHS_phi.index({k,j+1}), (r+0.5*dr)/dr2/r);
                }

                if (k < nz) {
                    P_phi.add(id, RHS_phi.index({k+1,j}), 1/dz2);
                }
            }
        }
        P_phi.close();
        P_phi.sort_rows();

        solver[i] = std::move(P_phi);
    }
};

}
