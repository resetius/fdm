#pragma once

#include "tensor.h"
#include "fft.h"
#include "asp_misc.h"
#include "verify.h"

namespace fdm {

struct LaplCyl3Data {
    const double dr, dz, dphi;
    const double dr2, dz2, dphi2;

    const double r0, lr, lz;
    const double slz;

    const int nr, nz, nphi;

    LaplCyl3Data(double dr, double dz,
                 double r0, double lr, double lz,
                 int nr, int nz, int nphi)
        : dr(dr), dz(dz), dphi(2*M_PI/nphi)
        , dr2(dr*dr), dz2(dz*dz), dphi2(dphi*dphi)
        , r0(r0), lr(lr), lz(lz), slz(sqrt(2./lz))
        , nr(nr), nz(nz), nphi(nphi)
    { }
};

template<typename T, template<typename> class Solver, bool check>
class LaplCyl3FFT1: public LaplCyl3Data {
public:
    constexpr static T SQRT_M_1_PI = 0.56418958354775629;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;
    using tensor = fdm::tensor<T,3,check,tensor_flags>;

    std::vector<int> indices;
    tensor RHS, ANS, RHSm, RHSx;
    std::vector<T> s, S;

    // strange
    std::vector<Solver<T>> solver;
    FFTTable<T> ft_table;
    std::vector<FFT<T>> ft;

    LaplCyl3FFT1(double dr, double dz,
             double r0, double lr, double lz,
             int nr, int nz, int nphi)
        : LaplCyl3Data(dr, dz, r0, lr, lz, nr, nz, nphi)
        , indices({0, nphi-1, 1, nz, 1, nr})
        , RHS(indices), ANS(indices), RHSm(indices), RHSx(indices)
        , s((nz+1)*nphi), S((nz+1)*nphi)
        , solver(nphi)
        , ft_table(nphi)
    {
        init_solver();
    }

    ~LaplCyl3FFT1()
    {
    }

    void solve(T* ans, T* rhs) {
        RHS.use(rhs); ANS.use(ans);

#pragma omp parallel for
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                for (int i = 0; i < nphi; i++) {
                    s[k*nphi+i] = RHS[i][k][j];
                }

                ft[k].pFFT_1(&S[k*nphi], &s[k*nphi], dphi*SQRT_M_1_PI);

                for (int i = 0; i < nphi; i++) {
                    RHSm[i][k][j] = S[k*nphi+i];
                }
            }
        }
#pragma omp parallel for
        for (int i = 0; i < nphi; i++) {
            // решаем nphi систем
            solver[i].solve(&RHSx[i][1][1], &RHSm[i][1][1]);
        }

#pragma omp parallel for
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                for (int i = 0; i < nphi; i++) {
                    s[k*nphi+i] = RHSx[i][k][j];
                }

                ft[k].pFFT(&S[k*nphi], &s[k*nphi], SQRT_M_1_PI);

                for (int i = 0; i < nphi; i++) {
                    ANS[i][k][j] = S[k*nphi+i];
                }
            }
        }
    }

private:
    void init_solver() {
        ft.reserve(nz+1);
        for (int i = 0; i < nphi; i++) {
            init_solver(i);
        }
        for (int k = 0; k <= nz; k++) {
            ft.emplace_back(ft_table, nphi);
        }
    }

    void init_solver(int i) {
        fdm::tensor<T,2,check> RHS_phi({1,nz,1,nr});
        csr_matrix<T> P_phi;

        T lm = 4.0/dphi2*asp::sq(sin(i*dphi*0.5));

        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                int id = RHS_phi.index({k,j});
                double r = r0+j*dr;

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

template<typename T, bool check>
class LaplCyl3FFT2: public LaplCyl3Data {
public:
    constexpr static T SQRT_M_1_PI = 0.56418958354775629;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;
    using tensor = fdm::tensor<T,3,check,tensor_flags>;

    std::vector<int> indices;
    tensor RHS, ANS, RHSm;

    int lds;
    std::vector<T> s, S;

    FFTTable<T> ft_phi_table;
    FFTTable<T> ft_z_table_;
    FFTTable<T>* ft_z_table;
    std::vector<FFT<T>> ft_phi;
    std::vector<FFT<T>> ft_z_;
    std::vector<FFT<T>>& ft_z;
    std::vector<T> lm_phi, lm_z;

    fdm::tensor<T,3,check> matrices;
    fdm::tensor<int,3,check> ipivs;

    LaplCyl3FFT2(double dr, double dz,
                 double r0, double lr, double lz,
                 int nr, int nz, int nphi)
        : LaplCyl3Data(dr, dz, r0, lr, lz, nr, nz, nphi)
        , indices({0, nphi-1, 1, nz, 1, nr})
        , RHS(indices), ANS(indices), RHSm(indices)
        , s((nz+1)*nphi), S((nz+1)*nphi)

        , ft_phi_table(nphi)
        , ft_z_table_(nphi == nz+1 ? 1 : nz+1)
        , ft_z_table(nphi == nz+1 ? &ft_phi_table : &ft_z_table_)

        , ft_z(nphi == nz+1 ? ft_phi : ft_z_)

        , lm_phi(nphi), lm_z(nz+1)
        , matrices({0,nphi-1,1,nz,0,4*nr-1})
        , ipivs({0,nphi-1,1,nz,0,nr-1})
    {
        init_solver();
    }

    ~LaplCyl3FFT2()
    {
    }

    void solve(T* ans, T* rhs) {
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

private:
    void init_solver() {
        for (int i = 0; i < nphi; i++) {
            lm_phi[i] = 4.0/dphi2*asp::sq(sin(i*dphi*0.5));
        }
        for (int k = 0; k <= nz; k++) {
            lm_z[k] = 4./dz2*asp::sq(sin(k*M_PI*0.5/(nz+1)));
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
