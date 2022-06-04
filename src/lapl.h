#pragma once

#include "tensor.h"
#include "asp_fft.h"
#include "verify.h"

namespace fdm {

template<typename T>
class LaplCyl3 {
    const double R, r0;
    const double h1, h2;

    const int nr, nz, nphi;
    const double dr, dz, dphi;
    const double dr2, dz2, dphi2;

public:
    LaplCyl3(double R, double r0, double h1, double h2,
             int nr, int nz, int nphi)
        : R(R), r0(r0)
        , h1(h1), h2(h2)
        , nr(nr), nz(nz), nphi(nphi)
        , dr((R-r0)/nr), dz((h2-h1)/nz), dphi(2*M_PI/nphi)
        , dr2(dr*dr), dz2(dz*dz), dphi2(dphi*dphi)
    { }

    void solve(double* ans, const double* rhs) {
        std::vector<int> indices = {0, nphi-1, 1, nz, 1, nr};
        tensor RHS(indices, rhs);
        tensor ANS(indices, ans);

        std::vector<T> s(nphi), S(nphi);

        fft* ft = FFT_init(FFT_PERIODIC, nphi);
        verify(ft);

        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                for (int i = 0; i < nphi; i++) {
                    s[i] = RHS[i][k][j];
                }

                pFFT_2(&S[0], &s[0], dphi, ft);
            }
        }

        FFT_free(ft);
    }
};

}
