#pragma once

#include "sparse.h"
#include "tensor.h"
#include "fft.h"
#include "vector.h"
#include "asp_misc.h"
#include "verify.h"

namespace fdm {

struct LaplCyl3Data {
    const double dr, dz, dphi;
    const double dr2, dz2, dphi2;

    const double r0, lr, lz;
    const double slz;

    const int zpoints;
    const int nr, nz, nphi;
    const int nrq;
    const int z1, zn;

    LaplCyl3Data(double dr, double dz,
                 double r0, double lr, double lz,
                 int nr, int nz, int nphi,
                 tensor_flag zflag = tensor_flag::none)
        : dr(dr), dz(dz), dphi(2*M_PI/nphi)
        , dr2(dr*dr), dz2(dz*dz), dphi2(dphi*dphi)
        , r0(r0), lr(lr), lz(lz), slz(sqrt(2./lz))
        , zpoints(zflag==tensor_flag::none?nz+1:nz)
        , nr(nr), nz(nz), nphi(nphi)
        , nrq(std::ceil(std::log2(nr+1)))
        , z1(zflag==tensor_flag::none?1:0)
        , zn(zflag==tensor_flag::none?nz:nz-1)
    { }
};

template<typename T, template<typename> class Solver, bool check>
class LaplCyl3FFT1: public LaplCyl3Data {
public:
    constexpr static T SQRT_M_1_PI = 0.56418958354775629;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;
    using tensor = fdm::tensor<T,3,check,tensor_flags>;

    std::array<int,6> indices;
    tensor RHS, ANS, RHSm, RHSx;
    OmpSafeTmpVector<T> s, S;

#ifdef HAVE_FFTW3
    using FFT_t = FFT_fftw3<T>;
#else
    using FFT_t = FFT<T>;
#endif

    // strange
    std::vector<Solver<T>> solver;
    FFTTable<T> ft_table;
    FFTOmpSafe<T,FFT_t> ft;

    LaplCyl3FFT1(double dr, double dz,
             double r0, double lr, double lz,
             int nr, int nz, int nphi)
        : LaplCyl3Data(dr, dz, r0, lr, lz, nr, nz, nphi)
        , indices({0, nphi-1, 1, nz, 1, nr})
        , RHS(indices), ANS(indices), RHSm(indices), RHSx(indices)
        , s(nphi), S(nphi)
        , solver(nphi)
        , ft_table(nphi)
#ifdef HAVE_FFTW3
        , ft(nphi)
#else
        , ft(ft_table, nphi)
#endif
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
                auto* in = s.data();
                auto* out = S.data();

                for (int i = 0; i < nphi; i++) {
                    in[i] = RHS[i][k][j];
                }

                ft.pFFT_1(out, in, dphi*SQRT_M_1_PI);

                for (int i = 0; i < nphi; i++) {
                    RHSm[i][k][j] = out[i];
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
                auto* in = s.data();
                auto* out = S.data();

                for (int i = 0; i < nphi; i++) {
                    in[i] = RHSx[i][k][j];
                }

                ft.pFFT(out, in, SQRT_M_1_PI);

                for (int i = 0; i < nphi; i++) {
                    ANS[i][k][j] = out[i];
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

template<typename T, bool check, tensor_flag zflag=tensor_flag::none, bool use_cyclic_reduction=false>
class LaplCyl3FFT2: public LaplCyl3Data {
public:
    constexpr static T SQRT_M_1_PI = 0.56418958354775629;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic,zflag>;
    using tensor = fdm::tensor<T,3,check,tensor_flags>;

    std::array<int,6> indices;
    tensor RHS, ANS, RHSm;

    int lds;
    OmpSafeTmpVector<T> s, S;

#ifdef HAVE_FFTW3
    using FFT_t = FFT_fftw3<T>;
#else
    using FFT_t = FFT<T>;
#endif

    FFTTable<T> ft_phi_table;
    FFTTable<T> ft_z_table_;
    FFTTable<T>* ft_z_table;
    FFTOmpSafe<T,FFT_t> ft_phi;
    FFTOmpSafe<T,FFT_t> ft_z_;
    FFTOmpSafe<T,FFT_t>& ft_z;
    std::vector<T> lm_phi, lm_z;

    fdm::tensor<T,3,check> matrices;
    fdm::tensor<int,3,check> ipivs;

    /**
       \param nr,nz,nphi - число ячеек по соответствующим направлениям
       \param dr,dz - размер ячеек
       \param r0 - координата первой точки по оси r
       (включая точку за границей области для смещенных сеток)
       \param lr, lz - расстояние от первой до последней точки
       (включая точки за границами).
       Для смещенных сеток lr=nr*dr+dr.
       Для обычных сеток lr=nr*dr.
     */
    LaplCyl3FFT2(double dr, double dz,
                 double r0, double lr, double lz,
                 int nr, int nz, int nphi)
        : LaplCyl3Data(dr, dz, r0, lr, lz, nr, nz, nphi, zflag)
        , indices({0, nphi-1, z1, zn, 1, nr})
        , RHS(indices), ANS(indices), RHSm(indices)
        , s(std::max(nphi, zn+1)), S(std::max(nphi, zn+1))

        , ft_phi_table(nphi)
        , ft_z_table_(nphi == zpoints ? 1 : zpoints)
        , ft_z_table(nphi == zpoints ? &ft_phi_table : &ft_z_table_)

#ifdef HAVE_FFTW3
        , ft_phi(nphi)
        , ft_z_(zpoints)
#else
        , ft_phi(ft_phi_table, nphi)
        , ft_z_(*ft_z_table, zpoints)
#endif

        , ft_z(nphi == zpoints ? ft_phi : ft_z_)

        , lm_phi(nphi), lm_z(zpoints)
        , matrices({0,nphi-1,z1,zn,0,4*nr-1})
        , ipivs({0,nphi-1,z1,zn,0,nr-1})
    {
        init_solver();
    }

    ~LaplCyl3FFT2()
    {
    }

    void solve(T* ans, T* rhs);

private:
    void init_solver();
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
        std::array<int,6> indices = {0, nphi-1, 1, nz, 1, nr};

        tensor<T,3,check,tensor_flags> RHS(indices, rhs);
        tensor<T,3,check,tensor_flags> ANS(indices, ans);

        solver.solve(&ANS[0][1][1], &RHS[0][1][1]);
    }

private:
    void init_solver() {
        using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

        std::array<int,6> indices = {0, nphi-1, 1, nz, 1, nr};
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
