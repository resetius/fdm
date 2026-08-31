#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>

#include "umfpack_solver.h"
#include "superlu_solver.h"
#include "lapl_cyl.h"
#include "config.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;
using namespace std;
using namespace std::chrono;
using namespace asp;

//(z-h0)*(z-h1)*(r-r0)*(r-R)*(sin(φ) + cos(φ))

double ans(int i, int k, int j, double dr, double dz, double dphi, double r0, double R, double h0, double h1) {
    double r = r0+dr*j-dr/2;
    double z = h0+dz*k-dz/2;
    double phi = dphi*(i+1)-dphi/2;
    double f = (z-h0)*(z-h1)*(r-r0)*(r-R)*(sin(phi) + cos(phi));
    return f;
}

//((r-r0)*(z-h0)*(z-h1)*(sin(φ)+cos(φ))+(r-R)*(z-h0)*(z-h1)*(sin(φ)+cos(φ))+2*r*(z-h0)*(z-h1)*(sin(φ)+cos(φ)))/r+2*(r-R)*(r-r0)*(sin(φ)+cos(φ))+((r-R)*(r-r0)*(z-h0)*(z-h1)*(-sin(φ)-cos(φ)))/r^2

double rp(int i, int k, int j, double dr, double dz, double dphi, double r0, double R, double h0, double h1) {
    double r = r0+dr*j-dr/2;
    double z = h0+dz*k-dz/2;
    double phi = dphi*(i+1)-dphi/2;
    double f = ((r-r0)*(z-h0)*(z-h1)*(sin(phi)+cos(phi))
                +(r-R)*(z-h0)*(z-h1)*(sin(phi)+cos(phi))
                +2*r*(z-h0)*(z-h1)*(sin(phi)+cos(phi)))/r
        +2*(r-R)*(r-r0)*(sin(phi)+cos(phi))
        +((r-R)*(r-r0)*(z-h0)*(z-h1)*(-sin(phi)-cos(phi)))/r/r;
    return f;
}

//(r-r0)*(r-R)*(sin(φ) + cos(φ)) + sin(z)^2+cos(z)^2
double ansp(int i, int k, int j, double dr, double dz, double dphi, double r0, double R, double h0, double h1) {
    double r = r0+dr*j-dr/2;
    double z = h0+dz*k-dz/2;
    double phi = dphi*(i+1)-dphi/2;
    double f = (r-r0)*(r-R)*(sin(phi) + cos(phi)) + sq(sin(z)) - sq(cos(z));
    return f;
}

double rpp(int i, int k, int j, double dr, double dz, double dphi, double r0, double R, double h0, double h1)
{
    double r = r0+dr*j-dr/2;
    double z = h0+dz*k-dz/2;
    double phi = dphi*(i+1)-dphi/2;

    return -4*sq(sin(z))+4*sq(cos(z))
        +((-sin(phi)-cos(phi))*(r-R)*(r-r0))/r/r
        +((sin(phi)+cos(phi))*(r-r0)
          +(sin(phi)+cos(phi))*(r-R)+(2*sin(phi)+2*cos(phi))*r)/r;
}

template<typename T,template<typename> class Solver>
void test_lapl_cyl_simple(void** data) {
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

    Config* c = static_cast<Config*>(*data);
    int nr = c->get("test", "nr", 16);
    int nz = c->get("test", "nz", 16);
    int nphi = c->get("test", "nphi", 16);
    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    LaplCyl3Simple<T, Solver, true> lapl(
        R, r0, h0, h1,
        nr, nz, nphi
        );
    double dr, dz, dphi;
    dr = lapl.dr; dz = lapl.dz; dphi = lapl.dphi;

    std::array<int,6> indices = {0, nphi-1, 1, nz, 1, nr};
    tensor<T,3,check,tensor_flags> RHS(indices);
    tensor<T,3,check,tensor_flags> ANS(indices);

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;

                RHS[i][k][j] = rp(i, k, j, dr, dz, dphi, r0, R, h0, h1);

                if (k <= 1) {
                    RHS[i][k][j] -= ans(i,k-1,j,dr,dz,dphi, r0, R, h0, h1)/dz/dz;
                }
                if (j <= 1) {
                    RHS[i][k][j] -= (r-dr/2)/r*ans(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r*ans(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (k >= nz) {
                    RHS[i][k][j] -= ans(i,k+1,j,dr,dz,dphi,r0,R,h0,h1)/dz/dz;
                }
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[0][1][1], &RHS[0][1][1]);
    auto t2 = steady_clock::now();

    double nrm = 0.0;
    double nrm1 = 0.0;
    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double f = ans(i,k,j, dr, dz, dphi,r0,R,h0,h1);
                if (verbose > 1) {
                    printf("%e %e %e %e\n",
                           ANS[i][k][j], f,
                           ANS[i][k][j]/f,
                           std::abs(ANS[i][k][j]-f));
                }

                nrm = std::max(nrm, std::abs(ANS[i][k][j]-f));
                nrm1 = std::max(nrm1, std::abs(f));
            }
        }
    }
    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

    assert_true(nrm < 1e-3);
}

void test_lapl_cyl_simple_double(void** data) {
    test_lapl_cyl_simple<double,umfpack_solver>(data);
}

void test_lapl_cyl_simple_float(void** data) {
    test_lapl_cyl_simple<float,superlu_solver>(data);
}

template<typename T,template<typename> class Solver>
void test_lapl_cyl(void** data) {
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

    int nr = c->get("test", "nr", 32);
    int nz = c->get("test", "nz", 31);
    int nphi = c->get("test", "nphi", 32);
    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    double dr = (R-r0)/nr;
    double dz = (h1-h0)/nz;
    double dphi = 2*M_PI/nphi;

    LaplCyl3FFT1<T, Solver, true> lapl(dr, dz, r0-dr/2, R-r0+dr, h1-h0+dz, nr, nz, nphi);

    std::array<int,6> indices = {0, nphi-1, 1, nz, 1, nr};
    tensor<T,3,check,tensor_flags> RHS(indices);
    tensor<T,3,check,tensor_flags> ANS(indices);

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;

                RHS[i][k][j] = rp(i, k, j, dr, dz, dphi, r0, R, h0, h1);

                if (k <= 1) {
                    RHS[i][k][j] -= ans(i,k-1,j,dr,dz,dphi, r0, R, h0, h1)/dz/dz;
                }
                if (j <= 1) {
                    RHS[i][k][j] -= (r-dr/2)/r*ans(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r*ans(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (k >= nz) {
                    RHS[i][k][j] -= ans(i,k+1,j,dr,dz,dphi,r0,R,h0,h1)/dz/dz;
                }
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[0][1][1], &RHS[0][1][1]);
    auto t2 = steady_clock::now();

    double nrm = 0.0;
    double nrm1 = 0.0;
    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double f = ans(i,k,j, dr, dz, dphi,r0,R,h0,h1);
                if (verbose > 1) {
                    printf("%e %e %e %e\n",
                           ANS[i][k][j], f,
                           ANS[i][k][j]/f,
                           std::abs(ANS[i][k][j]-f));
                }
                nrm = std::max(nrm, std::abs(ANS[i][k][j]-f));
                nrm1 = std::max(nrm1, std::abs(f));
            }
        }
    }
    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

    assert_true(nrm < 1e-3);
}

void test_lapl_cyl_double(void** data) {
    test_lapl_cyl<double,umfpack_solver>(data);
}

void test_lapl_cyl_float(void** data) {
    test_lapl_cyl<float,superlu_solver>(data);
}

template<typename T>
void test_lapl_cyl_zp(void** data) {
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic,tensor_flag::periodic>;

    int nr = c->get("test", "nr", 32);
    int nz = c->get("test", "nz", 32);
    int nphi = c->get("test", "nphi", 128);
    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = M_PI;
    double dr = (R-r0)/nr;
    double dz = (h1-h0)/nz;
    double dphi = 2*M_PI/nphi;

    LaplCyl3FFT2<T,true,tensor_flag::periodic>
        lapl(dr, dz, r0-dr/2, R-r0+dr, h1-h0, nr, nz, nphi);

    std::array<int,6> indices = {0, nphi-1, 0, nz-1, 1, nr};
    tensor<T,3,check,tensor_flags> RHS(indices);
    tensor<T,3,check,tensor_flags> ANS(indices);

    for (int i = 0; i < nphi; i++) {
        for (int k = 0; k < nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;

                RHS[i][k][j] = rpp(i, k, j, dr, dz, dphi, r0, R, h0, h1);

                if (j <= 1) {
                    RHS[i][k][j] -= (r-dr/2)/r*ansp(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r*ansp(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
            }
        }
    }

    auto t1 = steady_clock::now();
    //             phi z  r
    lapl.solve(&ANS[0][0][1], &RHS[0][0][1]);
    auto t2 = steady_clock::now();

    double nrm = 0.0;
    double nrm1 = 0.0;
    for (int i = 0; i < nphi; i++) {
        for (int k = 0; k < nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double f = ansp(i,k,j, dr, dz, dphi,r0,R,h0,h1);
                if (verbose > 1) {
                    printf("%e %e %e %e\n",
                           ANS[i][k][j], f,
                           ANS[i][k][j]/f,
                           std::abs(ANS[i][k][j]-f));
                }
                nrm = std::max(nrm, std::abs(ANS[i][k][j]-f));
                nrm1 = std::max(nrm1, std::abs(f));
            }
        }
    }
    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

    // RHS задан непрерывным оператором, поэтому остаётся ошибка O(h^2).
    assert_true(nrm < 1.5e-3);
}

void test_lapl_cyl_zp_double(void** data) {
    test_lapl_cyl_zp<double>(data);
}

void test_lapl_cyl_zp_float(void** data) {
    test_lapl_cyl_zp<float>(data);
}

template<typename T, tensor_flag zflag>
void test_lapl_cyl_fft2_discrete_impl(void** data, const int nr,
                                      const int nz, const int nphi) {
    constexpr bool check = true;
    constexpr bool periodic_z = zflag == tensor_flag::periodic;
    using tensor_flags = typename fdm::short_flags<tensor_flag::periodic, zflag>::value;

    Config* c = static_cast<Config*>(*data);
    int verbose = c->get("test", "verbose", 0);

    const int z1 = periodic_z ? 0 : 1;
    const int zn = periodic_z ? nz-1 : nz;
    const double r0 = 1.25;
    const double R = 2.75;
    const double lz = 2.0;
    const double dr = (R-r0)/nr;
    const double dz = lz/nz;
    const double dphi = 2*M_PI/nphi;

    LaplCyl3FFT2<T, check, zflag> lapl(
        dr, dz, r0-dr/2, R-r0+dr,
        periodic_z ? lz : lz+dz, nr, nz, nphi);

    std::array<int,6> indices = {0, nphi-1, z1, zn, 1, nr};
    tensor<T,3,check,tensor_flags> rhs(indices);
    tensor<T,3,check,tensor_flags> numerical(indices);

    auto exact = [&](int i, int k, int j) {
        const double radial = sin(M_PI*j/(nr+1));
        const double azimuthal = 1.0
            + 0.20*cos(2*M_PI*i/nphi)
            + 0.15*sin(4*M_PI*i/nphi);
        const double axial = periodic_z
            ? 1.0 + 0.25*cos(2*M_PI*k/nz) + 0.10*sin(4*M_PI*k/nz)
            : sin(M_PI*k/(nz+1));
        return radial*azimuthal*axial;
    };

    // RHS формируется тем же дискретным оператором; фиктивные значения равны нулю.
    for (int i = 0; i < nphi; i++) {
        for (int k = z1; k <= zn; k++) {
            for (int j = 1; j <= nr; j++) {
                const double r = r0+dr*j-dr/2;
                const double center = exact(i,k,j);
                rhs[i][k][j] =
                    ((r+dr/2)*exact(i,k,j+1)-2*r*center
                     +(r-dr/2)*exact(i,k,j-1))/(r*dr*dr)
                    +(exact(i,k+1,j)-2*center+exact(i,k-1,j))/(dz*dz)
                    +(exact(i+1,k,j)-2*center+exact(i-1,k,j))/(r*r*dphi*dphi);
            }
        }
    }

    lapl.solve(&numerical[0][z1][1], &rhs[0][z1][1]);

    double max_error = 0;
    double max_exact = 0;
    for (int i = 0; i < nphi; i++) {
        for (int k = z1; k <= zn; k++) {
            for (int j = 1; j <= nr; j++) {
                max_error = std::max(
                    max_error, std::abs(static_cast<double>(numerical[i][k][j])-exact(i,k,j)));
                max_exact = std::max(max_exact, std::abs(exact(i,k,j)));
            }
        }
    }
    const double relative_error = max_error/max_exact;
    if (verbose) {
        printf("FFT2 discrete inverse (%s, %s): err = %e\n",
               periodic_z ? "periodic z" : "Dirichlet z",
               std::is_same_v<T,double> ? "double" : "float",
               relative_error);
    }

    const double tolerance = std::is_same_v<T,double> ? 1e-11 : 2e-5;
    assert_true(relative_error < tolerance);
}

template<typename T, tensor_flag zflag>
void test_lapl_cyl_fft2_discrete(void** data) {
    constexpr bool periodic_z = zflag == tensor_flag::periodic;
    test_lapl_cyl_fft2_discrete_impl<T,zflag>(
        data, 9, periodic_z ? 8 : 7, 16);
}

void test_lapl_cyl_fft2_discrete_dirichlet_double(void** data) {
    test_lapl_cyl_fft2_discrete<double,tensor_flag::none>(data);
}

void test_lapl_cyl_fft2_discrete_dirichlet_float(void** data) {
    test_lapl_cyl_fft2_discrete<float,tensor_flag::none>(data);
}

void test_lapl_cyl_fft2_discrete_periodic_double(void** data) {
    test_lapl_cyl_fft2_discrete<double,tensor_flag::periodic>(data);
}

void test_lapl_cyl_fft2_discrete_periodic_float(void** data) {
    test_lapl_cyl_fft2_discrete<float,tensor_flag::periodic>(data);
}

void test_lapl_cyl_fft2_size_handling(void** data) {
#ifdef HAVE_FFTW3
    // Размеры 10 и 6 допустимы для FFTW, но не для встроенного radix-2 FFT.
    test_lapl_cyl_fft2_discrete_impl<double,tensor_flag::periodic>(
        data, 7, 6, 10);
#else
    bool rejected = false;
    try {
        LaplCyl3FFT2<double, true, tensor_flag::periodic> lapl(
            0.1, 0.2, 0.95, 0.8, 1.2, 7, 6, 10);
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("power-of-two") != std::string::npos;
    }
    assert_true(rejected);
#endif
}

template<typename T>
T solve_lapl(Config* c, int nr, int nz, int nphi) {
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    double dr = (R-r0)/nr;
    double dz = (h1-h0)/nz;
    double dphi = 2*M_PI/nphi;

    LaplCyl3FFT2<T, true> lapl(dr, dz, r0-dr/2, R-r0+dr, h1-h0+dz, nr, nz, nphi);

    std::array<int,6> indices = {0, nphi-1, 1, nz, 1, nr};
    tensor<T,3,check,tensor_flags> RHS(indices);
    tensor<T,3,check,tensor_flags> ANS(indices);

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;

                RHS[i][k][j] = rp(i, k, j, dr, dz, dphi, r0, R, h0, h1);

                if (k <= 1) {
                    RHS[i][k][j] -= ans(i,k-1,j,dr,dz,dphi, r0, R, h0, h1)/dz/dz;
                }
                if (j <= 1) {
                    RHS[i][k][j] -= (r-dr/2)/r*ans(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r*ans(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (k >= nz) {
                    RHS[i][k][j] -= ans(i,k+1,j,dr,dz,dphi,r0,R,h0,h1)/dz/dz;
                }
            }
        }
    }

    auto t1 = steady_clock::now();
    lapl.solve(&ANS[0][1][1], &RHS[0][1][1]);
    auto t2 = steady_clock::now();

    double nrm = 0.0;
    double nrm1 = 0.0;
    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double f = ans(i,k,j, dr, dz, dphi,r0,R,h0,h1);
                if (verbose > 1) {
                    printf("%e %e %e %e\n",
                           ANS[i][k][j], f,
                           ANS[i][k][j]/f,
                           std::abs(ANS[i][k][j]-f));
                }
                nrm = std::max(nrm, std::abs(ANS[i][k][j]-f));
                nrm1 = std::max(nrm1, std::abs(f));
            }
        }
    }
    nrm /= nrm1;
    auto interval = duration_cast<duration<double>>(t2 - t1);

    if (verbose) {
        printf("It took me '%f' seconds, err = '%e'\n", interval.count(), nrm);
    }

    return nrm;
}

template<typename T>
void test_lapl_cyl_norm_decr(void** data) {
    Config* c = static_cast<Config*>(*data);

    int nr = c->get("test", "nr", 16);
    int nz = c->get("test", "nz", 15);
    int nphi = c->get("test", "nphi", 16);

    double nrm1 = solve_lapl<T>(c, nr, nz, nphi);
    double nrm2 = solve_lapl<T>(c, nr*2, (nz+1)*2-1, nphi*2);
    int verbose = c->get("test", "verbose", 0);
    if (verbose) {
        printf("nrm1/nrm2 = %e, %e %e\n", nrm1/nrm2, nrm1, nrm2);
    }
    assert_true(nrm1/nrm2 > 3.7);
}

void test_lapl_cyl_norm_decr_double(void** data) {
    test_lapl_cyl_norm_decr<double>(data);
}

void test_lapl_cyl_norm_decr_float(void** data) {
    test_lapl_cyl_norm_decr<float>(data);
}

void test_lapl_cyl_fft1_fft2_cmp(void** data) {
    using T = double;
    Config* c = static_cast<Config*>(*data);
    constexpr bool check = true;
    using tensor_flags = fdm::tensor_flags<tensor_flag::periodic>;

    int nr = c->get("test", "nr", 32);
    int nz = c->get("test", "nz", 31);
    int nphi = c->get("test", "nphi", 32);
    int verbose = c->get("test", "verbose", 0);
    double r0 = M_PI/2, R = M_PI;
    double h0 = 0, h1 = 10;
    double dr = (R-r0)/nr;
    double dz = (h1-h0)/nz;
    double dphi = 2*M_PI/nphi;

    LaplCyl3FFT1<T, umfpack_solver, true> lapl1(dr, dz, r0-dr/2, R-r0+dr, h1-h0+dz, nr, nz, nphi);
    LaplCyl3FFT2<T, true> lapl2(dr, dz, r0-dr/2, R-r0+dr, h1-h0+dz, nr, nz, nphi);

    std::array<int,6> indices = {0, nphi-1, 1, nz, 1, nr};
    tensor<T,3,check,tensor_flags> RHS(indices);
    tensor<T,3,check,tensor_flags> ANS(indices);
    tensor<T,3,check,tensor_flags> ANS2(indices);

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                double r = r0+dr*j-dr/2;

                RHS[i][k][j] = rp(i, k, j, dr, dz, dphi, r0, R, h0, h1);

                if (k <= 1) {
                    RHS[i][k][j] -= ans(i,k-1,j,dr,dz,dphi, r0, R, h0, h1)/dz/dz;
                }
                if (j <= 1) {
                    RHS[i][k][j] -= (r-dr/2)/r*ans(i,k,j-1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (j >= nr) {
                    RHS[i][k][j] -= (r+dr/2)/r*ans(i,k,j+1,dr,dz,dphi,r0,R,h0,h1)/dr/dr;
                }
                if (k >= nz) {
                    RHS[i][k][j] -= ans(i,k+1,j,dr,dz,dphi,r0,R,h0,h1)/dz/dz;
                }
            }
        }
    }

    {
        auto t1 = steady_clock::now();
        lapl1.solve(&ANS[0][1][1], &RHS[0][1][1]);
        auto t2 = steady_clock::now();

        auto interval1 = duration_cast<duration<double>>(t2 - t1).count();
        if (verbose) {
            printf("%e\n", interval1);
        }
    }

    {
        auto t1 = steady_clock::now();
        lapl2.solve(&ANS2[0][1][1], &RHS[0][1][1]);
        auto t2 = steady_clock::now();

        auto interval1 = duration_cast<duration<double>>(t2 - t1).count();
        if (verbose) {
            printf("%e\n", interval1);
        }
    }

    for (int i = 0; i < nphi; i++) {
        for (int k = 1; k <= nz; k++) {
            for (int j = 1; j <= nr; j++) {
                assert_float_equal(ANS[i][k][j], ANS2[i][k][j], 1e-15);
            }
        }
    }
}

int main(int argc, char** argv) {
    string config_fn = "ut_lapl.ini";
    Config c;
    c.open(config_fn);
    c.rewrite(argc, argv);

    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_lapl_cyl_simple_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_simple_float, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_norm_decr_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_norm_decr_float, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_float, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_zp_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_zp_float, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_fft2_discrete_dirichlet_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_fft2_discrete_dirichlet_float, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_fft2_discrete_periodic_double, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_fft2_discrete_periodic_float, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_fft2_size_handling, &c),
        cmocka_unit_test_prestate(test_lapl_cyl_fft1_fft2_cmp, &c),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
