#pragma once

#include "tensor.h"
#include "fft.h"
#include "asp_misc.h"

namespace fdm {

template<typename T, bool check>
class LaplCube {
public:
    using tensor = tensor<T,3,check>;

    const double dx, dy, dz;
    const double dx2, dy2, dz2;

    const double lx, ly, lz;
    const double slx, sly, slz;

    const std::vector<int> indices;

    FFTTable<T> ft_y_table;
    FFT<T> ft_z;
    FFT<T> ft_y;
    FFT<T> ft_x;

    std::vector<T> lm_y;
    std::vector<T> lm_x_;
    std::vector<T> lm_z_;
    T* lm_x;
    T* lm_z;

    LaplCube(double dx, double dy, double dz,
             double lx, double ly, double lz,
             int nx, int ny, int nz)
        : dx(dx), dy(dy), dz(dz)
        , dx2(dx*dx), dy2(dy*dy), dz2(dz*dz)
        , lx(lx), ly(ly), lz(lz)
        , slx(sqrt(2./lx)), sly(sqrt(2./ly)), slz(sqrt(2./lz))
        , indices({1,nz,1,ny,1,nz})

        , ft_y_table(ny) // TODO
        , ft_z(ft_y_table, nz) // TODO
        , ft_y(ft_y_table, ny) // TODO
        , ft_x(ft_x_table, nx) // TODO
    {
        init_lm();
    }

    void solve(T* ans, T* rhs) {
        tensor ANS(indices, ans);
        tensor RHS(indices, rhs);
        tensor RHSm(indices);

        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                for (int i = 1; i <= nz; i++) {
                    s[i] = RHS[i][k][j];
                }

                ft_z.sFFT(&S[0], &s[0], dz*slz);

                for (int i = 1; i <= nz; i++) {
                    RHSm[i][k][j] = S[i];
                }
            }
        }

        for (int i = 1; i <= nz; i++) {
            for (int j = 1; j <= nx; j++) {
                for (int k = 1; k <= ny; k++) {
                    s[k] = RHSm[i][k][j];
                }

                ft_y.sFFT(&S[0], &s[0], dy*sly);

                for (int k = 1; k <= ny; k++) {
                    RHSm[i][k][j] = S[k];
                }
            }
        }

        for (int i = 1; i <= nz; i++) {
            for (int k = 1; k <= ny; k++) {
                for (int j = 1; j <= nx; j++) {
                    s[j] = RHSm[i][k][j];
                }

                ft_x.sFFT(&S[0], &s[0], dx*slx);

                for (int j = 1; j <= nx; j++) {
                    RHSm[i][k][j] = S[j];
                }
            }
        }

        for (int i = 1; i <= nz; i++) {
            for (int k = 1; k <= ny; k++) {
                for (int j = 1; j <= nx; j++) {
                    RHSm[k][j] /= -lm_z[i]-lm_y[k]-lm_x[j];
                }
            }
        }

        for (int i = 1; i <= nz; i++) {
            for (int k = 1; k <= ny; k++) {
                for (int j = 1; j <= nx; j++) {
                    s[j] = RHSm[i][k][j];
                }

                ft_x.sFFT(&S[0], &s[0], dx*slx);

                for (int j = 1; j <= nx; j++) {
                    ANS[i][k][j] = S[j];
                }
            }
        }

        for (int i = 1; i <= nz; i++) {
            for (int j = 1; j <= nx; j++) {
                for (int k = 1; k <= ny; k++) {
                    s[k] = ANS[i][k][j];
                }

                ft_y.sFFT(&S[0], &s[0], dy*sly);

                for (int k = 1; i <= ny; k++) {
                    ANS[i][k][j] = S[k];
                }
            }
        }

        for (int k = 1; k <= ny; k++) {
            for (int j = 1; j <= nx; j++) {
                for (int i = 1; i <= nz; i++) {
                    s[i] = ANS[i][k][j];
                }

                ft_z.sFFT(&S[0], &s[0], dz*slz);

                for (int i = 1; i <= nz; i++) {
                    ANS[i][k][j] = S[i];
                }
            }
        }
    }

private:
    void init_lm() {
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
};

} // namespace fdm
