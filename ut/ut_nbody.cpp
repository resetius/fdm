#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>
#include <random>

#include "fft.h"
#include "asp_misc.h"
#include "tensor.h"
#include "interpolate.h"

extern "C" {
#include <cmocka.h>
}

using namespace fdm;
using namespace std;
using namespace std::chrono;
using namespace asp;

struct Body {
    double x[3];
    double a[3];
    double F[3];
    double mass;
};

void test_poor_man_poisson(void** ) {
    constexpr auto flag = tensor_flag::periodic;
    using T = double;
    using flags3 = typename short_flags<flag,flag,flag>::value;
    using tensor = fdm::tensor<T,3,true,flags3>;
    using tensor4 = fdm::tensor<T,4,true,flags3>;

    T l = 2*M_PI;
    int n = 128;
    tensor rhs({0,n-1,0,n-1,0,n-1});
    tensor psi({0,n-1,0,n-1,0,n-1});
    tensor4 E({0,n-1,0,n-1,0,n-1,0,3});
    vector<T> S(n), s(n);

    int N = 20;
    vector<Body> bodies(N);
    T h = l/n;

    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(0, l/8);

    for (int index = 0; index < N; index++) {
        auto& body = bodies[index];
        for (int i = 0; i < 3; i++) {
            body.x[i] = distribution(generator) + l/4;
        }
        body.mass = 0.2 + 1.5 * distribution(generator) / l;
    }

    CIC3<T> int1;

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                rhs[i][k][j] = 0;
            }
        }
    }

    for (auto& body : bodies) {
        CIC3<T>::matrix M;
        int i0, j0, k0;
        int1.distribute(
            M, body.x[0], body.x[1], body.x[2],
            &j0, &k0, &i0, h
            );

        for (int i = 0; i < CIC3<T>::n; i++) {
            for (int k = 0; k < CIC3<T>::n; k++) {
                for (int j = 0; j < CIC3<T>::n; j++) {
                    rhs[i+i0][k+k0][j+j0] += -4*M_PI*body.mass*M[i][k][j]/h/h/h;
                }
            }
        }
    }

    FFTTable<T> ft_table(n);
    FFT<T> ft(ft_table, n);
    T slh = sqrt(2./l);

    for (int k = 0; k < n; k++) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                s[i] = rhs[i][k][j];
            }

            ft.pFFT_1(&S[0], &s[0], h*slh);

            for (int i = 0; i < n; i++) {
                rhs[i][k][j] = S[i];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                s[k] = rhs[i][k][j];
            }

            ft.pFFT_1(&S[0], &s[0], h*slh);

            for (int k = 0; k < n; k++) {
                rhs[i][k][j] = S[k];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                s[j] = rhs[i][k][j];
            }

            ft.pFFT_1(&S[0], &s[0], h*slh);

            for (int j = 0; j < n; j++) {
                rhs[i][k][j] = S[j];
            }
        }
    }

    ///
    /// -4piG ro /k/k exp (-k^2 r^2)
    ///

    vector<T> lm(n);
    for (int k = 0; k < n; k++) {
        lm[k] = 4./h/h*sq(sin(k*M_PI/(n)));
    }

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                /*T kk[3];
                kk[0] = j<n/2?(T)j/n:- (T)(n-(j-n/2.))/n;
                kk[1] = k<n/2?(T)k/n:- (T)(n-(k-n/2.))/n;
                kk[2] = i<n/2?(T)i/n:- (T)(n-(i-n/2.))/n;
                for (int q = 0; q < 3; q++) {
                    kk[q] *= 4/h/h;
                }
                rhs[i][k][j] *= 4*M_PI / (sq(kk[0])+sq(kk[1])+sq(kk[2]));
                */
                rhs[i][k][j] /= -(lm[i]+lm[k]+lm[j]);
            }
        }
    }

    rhs[0][0][0] = 1;

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                s[j] = rhs[i][k][j];
            }

            ft.pFFT(&S[0], &s[0], slh);

            for (int j = 0; j < n; j++) {
                psi[i][k][j] = S[j];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                s[k] = psi[i][k][j];
            }

            ft.pFFT(&S[0], &s[0], slh);

            for (int k = 0; k < n; k++) {
                psi[i][k][j] = S[k];
            }
        }
    }

    for (int k = 0; k < n; k++) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                s[i] = psi[i][k][j];
            }

            ft.pFFT(&S[0], &s[0], slh);

            for (int i = 0; i < n; i++) {
                psi[i][k][j] = S[i];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                E[i][k][j][0] = -(psi[i][k][j+1]-psi[i][k][j-1])/2/h;
                E[i][k][j][1] = -(psi[i][k+1][j]-psi[i][k-1][j])/2/h;
                E[i][k][j][2] = -(psi[i+1][k][j]-psi[i-1][k][j])/2/h;
            }
        }
    }

    for (auto& body : bodies) {
        T a[3] = {0};
        CIC3<T>::matrix M;
        int i0, j0, k0;
        int1.distribute(
            M, body.x[0], body.x[1], body.x[2],
            &j0, &k0, &i0, h
            );
        for (int i = 0; i < CIC3<T>::n; i++) {
            for (int k = 0; k < CIC3<T>::n; k++) {
                for (int j = 0; j < CIC3<T>::n; j++) {
                    for (int m = 0; m < 3; m++) {
                        a[m] += E[i0+i][k0+k][j0+j][m] * M[i][k][j];
                    }
                }
            }
        }

        for (int i = 0; i < 3; i++) {
            body.a[i] = a[i];
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            auto& bi = bodies[i];
            auto& bj = bodies[j];

            double R = 0;
            for (int k = 0; k < 3; k++) {
                R += sq(bi.x[k]-bj.x[k]);
            }
            R = std::sqrt(R);
            for (int k = 0; k < 3; k++) {
                bi.F[k] +=  bj.mass * (bi.x[k] - bj.x[k]) /R/R/R;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        auto& bi = bodies[i];
        double R = 0, RR = 0;
        for (int k = 0; k < 2; k++) {
            R += sq(bi.a[k]-bi.F[k]);
            RR += sq(bi.F[k]);
        }
        R = std::sqrt(R); RR=std::sqrt(RR);
        R /= RR;
        //printf("< %e %e %e\n", bi.mass, bi.x[0], bi.x[1]);
        printf("> %e %+e %+e %+e %+e\n", R, bi.a[0], bi.F[0], bi.a[1], bi.F[1]);
    }
}

int main(int argc, char** argv) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_poor_man_poisson, nullptr),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
