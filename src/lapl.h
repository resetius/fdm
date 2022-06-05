#pragma once

#include "tensor.h"
#include "asp_fft.h"
#include "asp_misc.h"
#include "verify.h"

namespace fdm {

template<typename T, template<typename> class Solver, bool check>
class LaplCyl3 {
public:
    constexpr static T SQRT_M_1_PI = 0.56418958354775629;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;
    using tensor = fdm::tensor<T,3,check,tensor_flags>;

    const double R, r0;
    const double h1, h2;

    const int nr, nz, nphi;
    const double dr, dz, dphi;
    const double dr2, dz2, dphi2;

    std::vector<int> indices;

    // strange
    std::vector<Solver<T>> solver;
    fft* ft;

    LaplCyl3(double R, double r0, double h1, double h2,
             int nr, int nz, int nphi)
        : R(R), r0(r0)
        , h1(h1), h2(h2)
        , nr(nr), nz(nz), nphi(nphi)
        , dr((R-r0)/nr), dz((h2-h1)/nz), dphi(2*M_PI/nphi)
        , dr2(dr*dr), dz2(dz*dz), dphi2(dphi*dphi)
        , indices({0, nphi-1, 1, nz, 1, nr})
        , solver(nphi)
        , ft(FFT_init(FFT_PERIODIC, nphi))
    {
        init_solver();
        verify(ft);
    }

    ~LaplCyl3() {
        FFT_free(ft);
    }

    void solve(T* ans, T* rhs) {
        tensor RHS(indices, rhs);
        tensor ANS(indices, ans);
        tensor RHSm(indices);
        tensor RHSx(indices);

        std::vector<T> s(nphi), S(nphi);

        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                for (int i = 0; i < nphi; i++) {
                    s[i] = RHS[i][k][j];
                }

                pFFT_2_1(&S[0], &s[0], dphi*SQRT_M_1_PI, ft);

                for (int i = 0; i < nphi; i++) {
                    RHSm[i][k][j] = S[i];
                }
            }
        }
#pragma omp parallel for
        for (int i = 0; i < nphi; i++) {
            // решаем nphi систем
            solver[i].solve(&RHSx[i][1][1], &RHSm[i][1][1]);
        }

        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                for (int i = 0; i < nphi; i++) {
                    s[i] = RHSx[i][k][j];
                }

                pFFT_2(&S[0], &s[0], SQRT_M_1_PI, ft);

                for (int i = 0; i < nphi; i++) {
                    ANS[i][k][j] = S[i];
                }
            }
        }
    }

private:
    void init_solver() {
        for (int i = 0; i < nphi; i++) {
            init_solver(i);
        }
    }

    void init_solver(int i) {
        fdm::tensor<T,2,check> RHS_phi({1,nz,1,nr});
        csr_matrix<T> P_phi;

        T lm = 4.0/dphi2*asp::sq(sin(i*dphi*0.5));

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

                P_phi.add(id, RHS_phi.index({k,j}), -2/dr2-2/dz2-lm/r/r);

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


template<typename T, template<typename> class Solver, bool check>
class LaplCyl3Simple {
public:

    const double R, r0;
    const double h1, h2;

    const int nr, nz, nphi;
    const double dr, dz, dphi;
    const double dr2, dz2, dphi2;

    Solver<T> solver;

    LaplCyl3Simple(double R, double r0, double h1, double h2,
             int nr, int nz, int nphi)
        : R(R), r0(r0)
        , h1(h1), h2(h2)
        , nr(nr), nz(nz), nphi(nphi)
        , dr((R-r0)/nr), dz((h2-h1)/nz), dphi(2*M_PI/nphi)
        , dr2(dr*dr), dz2(dz*dz), dphi2(dphi*dphi)
    {
        init_solver();
    }

    void solve(T* ans, T* rhs) {
        using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;
        std::vector<int> indices = {0, nphi-1, 1, nz, 1, nr};

        tensor<T,3,check,tensor_flags> RHS(indices, rhs);
        tensor<T,3,check,tensor_flags> ANS(indices, ans);

        solver.solve(&ANS[0][1][1], &RHS[0][1][1]);
    }

private:
    void init_solver() {
        using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

        std::vector<int> indices = {0, nphi-1, 1, nz, 1, nr};
        tensor<T,3,check,tensor_flags> RHS(indices);

        csr_matrix<T> P;

        for (int i = 0; i < nphi; i++) {
            for (int k = 1; k <= nz; k++) {
                for (int j = 1; j <= nr; j++) {
                    int id = RHS.index({i,k,j});
                    double r = r0+j*dr-dr/2;
                    double r2 = r*r;
                    //double rm1 = (r-0.5*dr)/r;
                    //double rm2 = (r+0.5*dr)/r;
                    double rm = 0;
                    double zm = 0;

                    /*if (j > 1)  { rm += rm1; }
                    if (j < nr) { rm += rm2; }
                    if (k > 1)  { zm += 1; }
                    if (k < nz) { zm += 1; }*/
                    zm = 2; rm = 2;

                    P.add(id, RHS.index({i-1,k,j}), 1/dphi2/r2);

                    if (k > 1) {
                        P.add(id, RHS.index({i,k-1,j}), 1/dz2);
                    }

                    if (j > 1) {
                        P.add(id, RHS.index({i,k,j-1}), (r-0.5*dr)/dr2/r);
                    }

                    P.add(id, RHS.index({i,k,j}), -rm/dr2-zm/dz2-2/dphi2/r2);

                    if (j < nr) {
                        P.add(id, RHS.index({i,k,j+1}), (r+0.5*dr)/dr2/r);
                    }

                    if (k < nz) {
                        P.add(id, RHS.index({i,k+1,j}), 1/dz2);
                    }

                    P.add(id, RHS.index({i+1,k,j}), 1/dphi2/r2);
                }
            }
        }
        P.close();
        P.sort_rows();

        solver = std::move(P);
    }
};

}
