#include <cmath>
#include <complex>
#include <string.h>
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "fft.h"
#include "verify.h"

using namespace std;

namespace fdm {

#define _2(a) (1<<(a))

template<typename T>
void FFTTable<T>::init() {
    int n1 = N;
    int n = 0;
    while (n1 % 2 == 0) {
        n1 /= 2; n++;
    }

    verify((1<<n) == N);

    ffCOS.resize(2*N);
    ffSIN.resize(2*N);
    ffiCOS.resize((n+1)*N);

    for (int m = 0; m < 2*N; ++m) {
        ffCOS[m] = cos(m * M_PI/N);
        ffSIN[m] = sin(m * M_PI/N);
    }

    ffEXPr.resize(n);
    ffEXPi.resize(n);
    for (int s = 1; s <= n; s++) {
        int m = 1<<s;
        ffEXPr[s-1] = cos(-2*M_PI/m);
        ffEXPi[s-1] = sin(-2*M_PI/m);
    }

#define off(k,l) ((l+1)*N+(k-1))
    for (int l = -1; l <= n-1; l++) {
        for (int k = 1; k <= _2(l+1); k++) {
            T x = 2.*cos(M_PI*(2*k-1)/(_2(l+2)));
            x = 1./x;
            ffiCOS[off(k,l)] = x;
        }
    }
#undef off
}

template<typename T>
void FFT<T>::init() {
    int n1 = N;
    n = 0;
    while (n1 % 2 == 0) {
        n1 /= 2; n++;
    }

    verify((1<<n) == N);
}

template<typename T>
inline void FFT<T>::padvance(T* a, int idx)
{
    for (int j = 0; j <= idx - 1; j++) {
        T a1, a2;
        a1 = a[j] + a[idx + j];
        a2 = a[j] - a[idx + j];
        a[j]         = a1;
        a[idx + j] = a2;
    }
}

template<typename T>
void FFT<T>::pFFT_1_old(T *S, T *s1, T dx) {
    const T* ffSIN = &t.ffSIN[0];
    const T* ffCOS = &t.ffCOS[0];

    int s, j, idx, idx2, vm, k;
    int sz = static_cast<int>(t.ffSIN.size());
    int N_2 = N/2;
    int yoff = 0;
    int _yoff = N_2;

    T* a = s1;

    for (s = 1; s <= n - 2; s++) {
        idx = 1 << (n - s - 1);
        vm  = 1 << s;

        padvance(a, 2*idx);

        for (k = 1; k <= idx; k++) {
            T s1 = 0.0;
            T s2 = 0.0;
            for (j = 0; j <= 2 * idx - 1; j++) {
                s1 += a[2 * idx + j] *
                    ffCOS[((2 * k - 1) * vm * j) % sz];
            }
            for (j = 1; j <= 2 * idx - 1; j++) {
                s2 += a[2 * idx + j] *
                    ffSIN[((2 * k - 1) * vm * j) % sz];
            }
            idx2 = (1 << (s - 1)) * (2 * k - 1);
            S[yoff + idx2]  = s1;
            S[N - idx2] = s2;
        }
    }


    padvance(a, 1 << (n - (n-1)));
    S[yoff +(1 << (n - 2))] = a[2];
    S[N-(1 << (n - 2))] = a[3];

    padvance(a, 1 << (n - n));
    S[yoff + 0]             = a[0];
    S[yoff + N_2]           = a[1];

    for (k = 0; k <= N_2; k++) {
        S[yoff + k]  = S[yoff + k] * dx;
    }

    for (int k = 1; k <= N_2-1; k++) {
        S[_yoff + k] = S[_yoff + k] * dx;
    }
}

template<typename T>
void FFT<T>::pFFT_1(T *S, T *s1, T dx) {
    std::vector<T> b(N); // remove me
    std::vector<T> bn(N); // remove me
    std::vector<T>& z = b;
    std::vector<T>& zn = bn;

    int N_2 = N/2;
    int yoff = 0;
    int _yoff = N_2;

    int s, k, l, m, j;
    T* a = s1;

#define off(a,b) ((a)*(_2(m))+(b-1))
#define _off(a,b) ((a)*(_2(m-1))+(b-1))

#define zoff(a,b) ((a-1)*(_2(m))+(b-1))
#define _zoff(a,b) ((a-1)*(_2(m-1))+(b-1))

    for (l = n-1; l >= 2; l--) {
        // l = n-s
        padvance(a, _2(l));

        m = 0;
        for (j = 0; j <= _2(l)-1; j++) {
            b[off(j,1)] = a[_2(l)+j];
        }

        for (m = 1; m <= l-1; m++) {
            for (s = 1; s <= _2(m-1); s++) {
                j = 0;
                bn[off(j,2*s-1)] = b[_off(1,s)] - b[_off(_2(l-m+1)-1,s)];
                bn[off(j,2*s)] = b[_off(2*j,s)];

                for (j = 1; j <= _2(l-m)-1; j++) {
                    bn[off(j,2*s-1)] = b[_off(2*j-1,s)] + b[_off(2*j+1,s)];
                    bn[off(j,2*s)] = b[_off(2*j,s)];
                }
            }
            bn.swap(b);
        }
        m=l-1;

        for (s = 1; s <= _2(l-1); s++) {
            zn[zoff(1,s)] = b[off(0,s)];
            zn[_yoff+zoff(1,s)] = b[off(1,s)];
        }
        zn.swap(z);

        for (m = l-1; m >= 1; m--) {
            for (k = 1; k <= _2(l-m-1); k++) {
                for (s = 1; s <= _2(m-1); s++) {
                    zn[_zoff(k,s)] = z[zoff(k,2*s)]
                        + t.iCOS(k,l-m-1)*z[zoff(k,2*s-1)];
                    zn[_zoff(_2(l-m)-k+1,s)] = z[zoff(k,2*s)]
                        - t.iCOS(k,l-m-1)*z[zoff(k,2*s-1)];

                    zn[_yoff+_zoff(k,s)] = z[_yoff+zoff(k,2*s)]
                        + t.iCOS(k,l-m-1)*z[_yoff+zoff(k,2*s-1)];
                    zn[_yoff+_zoff(_2(l-m)-k+1,s)] = -z[_yoff+zoff(k,2*s)]
                        + t.iCOS(k,l-m-1)*z[_yoff+zoff(k,2*s-1)];
                }
            }
            zn.swap(z);
        }

        for (k = 1; k <= _2(l-1); k++) {
            S[yoff+_2(n-l-1)*(2*k-1)] = z[zoff(k,1)];
            S[N-(_2(n-l-1)*(2*k-1))] = z[_yoff+zoff(k,1)];
        }
    }

    padvance(a, 1 << (n - (n-1)));
    S[yoff +(1 << (n - 2))] = a[2];
    S[N-(1 << (n - 2))] = a[3];

    padvance(a, 1 << (n - n));
    S[yoff + 0]             = a[0];
    S[yoff + N_2]           = a[1];

    for (k = 0; k <= N_2; k++) {
        S[yoff + k]  = S[yoff + k] * dx;
    }

    for (k = 1; k <= N_2-1; k++) {
        S[_yoff + k] = S[_yoff + k] * dx;
    }

#undef off
#undef _off
#undef zoff
#undef _zoff
}

template<typename T>
void FFT<T>::pFFT(T *S, T* s, T dx) {
    int N_2 = N/2;
    int k;

    //cFFT_old(&S[0], &s[0], dx, N_2,n-1,2);
    cFFT(&S[0], &s[0], dx, N_2,n-1);

    // S[N_2] not filled, N_2+1 first non empty
    //sFFT_old(&s[0], &s[N_2], dx, N_2,n-1,2);
    sFFT(&s[0], &s[N_2], dx, N_2,n-1);

    for (k = 1; k <= N_2 - 1; k ++) {
        int r = k%2;
        T S_k   = (S[k] + (2*r-1)*s[k]);
        T S_N_k = (S[k] - (2*r-1)*s[k]);
        S[k]    = S_k;
        S[N-k]  = S_N_k;
    }
}

template<typename T>
void FFT<T>::sFFT_old(T *S,T *s,T dx) {
    sFFT_old(S, s, dx, N, n, 1);
}

template<typename T>
inline void FFT<T>::sadvance(T*a, int idx) {
    for (int j = 1; j <= idx - 1; j ++) {
        T a1 = a[j] - a[2 * idx - j];
        T a2 = a[j] + a[2 * idx - j];
        a[j]           = a1;
        a[2 * idx - j] = a2;
    }
}

template<typename T>
void FFT<T>::sFFT_old(T *S,T *s,T dx,int N,int n,int nr) {
    const T* ffSIN = &t.ffSIN[0];
    int sz = static_cast<int>(t.ffSIN.size());

    T*a = s;

    for (int s = 1; s <= n - 1; s++) {
        int idx = 1 << (n - s);
        int vm  = 1 << (s - 1);
        sadvance(a,idx);

        for (int k = 1; k <= idx; k++) {
            T y = 0;
            for (int j = 1; j <= idx; j++) {
                y += a[idx * 2 - j] *
                    ffSIN[((2 * k - 1) * vm * nr * j) % sz];

            }
            S[(2 * k - 1) * vm] = y * dx;
        }
    }
    int idx = 1 << (n - 1);
    S[idx] = a[1] * dx;
}

template<typename T>
void prn(T*a, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f ", a[i]);
    }
    printf("\n\n");
}

template<typename T>
void FFT<T>::sFFT(T* S, T* s, T dx, int N, int n) {
    std::vector<T> b(N); // remove me
    std::vector<T> bn(N); // remove me
    std::vector<T>& z = b;
    std::vector<T>& zn = bn;

    T* a = s;
    // s = 1,...,2^{m-1}
#define off(a,b) ((a-1)*(_2(m))+(b-1))
#define _off(a,b) ((a-1)*(_2(m-1))+(b-1))
    for (int l = n-1; l >= 1; l--) { // l=n-s
        // (30), p 170
        sadvance(a, 1<<l); // a^s

        // (36), p 172
        // b^0 = a
        int m = 0, j = 0, s = 0, k = 0;
        for (j = 1; j <= _2(l-m); j++) {
            b[off(j,1)] = a[(_2(l+1))-j]; // (m=0)
        }
        // b^m, m = 1, ... l, incr
        for (m = 1; m <= l-1; m++) {
            for (s = 1; s <= _2(m-1); s++) {
                for (j = 1; j <= _2(l-m)-1; j++) {
                    bn[off(j,2*s-1)] = b[_off(2*j-1,s)]+b[_off(2*j+1,s)];
                    bn[off(j,2*s)]   = b[_off(2*j,s)];
                }
                // j = _2(l-m)
                bn[off(j,2*s-1)] = b[_off(_2(l-m+1)-1, s)];
                bn[off(j,2*s)]   = b[_off(2*j,s)];
            }
            bn.swap(b);
        }
        // m = l
        for (s = 1; s <= _2(m-1); s++) {
            bn[off(1,2*s)] = b[_off(2,s)];
            bn[off(1,2*s-1)] = b[_off(1,s)];
        }
        bn.swap(b);

        // (37), p 172
        // z^l = b^l
        // z^m, m = l, ...,0, decr
        for (m = l; m >= 1; m--) {
            for (k = 1; k <= _2(l-m); k++) {
                for (s = 1; s <= _2(m-1); s++) {
                    zn[_off(k,s)] = z[off(k,2*s)]
                        + t.iCOS(k,l-m)*z[off(k,2*s-1)];
                    zn[_off(_2(l-m+1)-k+1,s)] = -z[off(k,2*s)]
                        + t.iCOS(k,l-m)*z[off(k,2*s-1)];
                }
            }
            zn.swap(z);
        }
        // z^0 -> y (ans)
        for (k = 1; k <= _2(l); k++) {
            S[(_2(n-l-1))*(2*k-1)] = dx*z[off(k,1)];
        }
    }

    // (31), p 170
    S[_2(n-1)] = dx*a[1];

#undef off
#undef _off
}

#ifdef _OPENMP
template<typename T>
inline void sadvance_omp(T* a, int idx, int id, int work) {
    for (int j = id+1; j < id+1+work && j <= idx-1; j++) {
        T a1 = a[j] - a[2 * idx - j];
        T a2 = a[j] + a[2 * idx - j];
        a[j]           = a1;
        a[2 * idx - j] = a2;
    }
}
#endif

template<typename T>
void FFT<T>::sFFT_omp(T* S, T* s, T dx) {
#ifdef _OPENMP
    std::vector<T> b(2*N); // remove me
    std::vector<T> bn(2*N); // remove me
    std::vector<T>& z = b;
    std::vector<T>& zn = bn;

    T* a = s;
    // s = 1,...,2^{m-1}
#define off(a,b) ((a-1)*(_2(m))+(b-1))
#define _off(a,b) ((a-1)*(_2(m-1))+(b-1))

    int size = 4; // _2(n-1);

    omp_set_dynamic(0);
    omp_set_num_threads(size);

#pragma omp parallel
    {
    int thread_id = omp_get_thread_num();
    for (int l = n-1; l >= 1; l--) { // l=n-s
        int work = std::max(1, _2(l) / size);
        int id = thread_id*work;

        sadvance_omp(a, _2(l), id, work);
#pragma omp barrier

        int m = 0, j = 0, s = 0, k = 0;
        for (j = id+1; j < id+1+work && j <= _2(l); j++) {
            b[off(j,1)] = a[(_2(l+1))-j];
        }
#pragma omp barrier

        for (m = 1; m <= l-1; m++) {
            int ns = _2(m-1);
            int nj = _2(l-m);
            for (int i = id; i < id+work; i++) {
                j = i/ns + 1;
                s = i%ns + 1;
                if (j <= nj) {
                    if (j == nj) {
                        bn[off(j,2*s-1)] = b[_off(2*j-1,s)];
                    } else {
                        bn[off(j,2*s-1)] = b[_off(2*j-1,s)]+b[_off(2*j+1,s)];
                    }
                    bn[off(j,2*s)]   = b[_off(2*j,s)];
                }
            }
#pragma omp barrier
            if (id == 0) bn.swap(b);
#pragma omp barrier
        }
#pragma omp barrier

        int ns = _2(m-1);
        for (s = id + 1; s < id+1+work && s <= ns; s++) {
            bn[off(1,2*s)] = b[_off(2,s)];
            bn[off(1,2*s-1)] = b[_off(1,s)];
        }
#pragma omp barrier
        if (id == 0) bn.swap(b);
#pragma omp barrier

        for (m = l; m >= 1; m--) {
            int ns = _2(m-1);
            int nk = _2(l-m);
            for (int i = id; i < id+work; i++) {
                k = i/ns + 1;
                s = i%ns + 1;

                if (k <= nk) {
                    zn[_off(k,s)] = z[off(k,2*s)]
                        + t.iCOS(k,l-m)*z[off(k,2*s-1)];
                    zn[_off(_2(l-m+1)-k+1,s)] = -z[off(k,2*s)]
                        + t.iCOS(k,l-m)*z[off(k,2*s-1)];
                }
            }
#pragma omp barrier
            if (id == 0) zn.swap(z);
#pragma omp barrier
        }
        for (k = id+1; k < id+1+work&&k <= _2(l); k++) {
            S[(_2(n-l-1))*(2*k-1)] = dx*z[off(k,1)];
        }

#pragma omp barrier
    }
    }

    S[_2(n-1)] = dx*a[1];

#undef off
#undef _off

#else
    abort();
#endif
}

template<typename T>
void FFT<T>::sFFT(T* S, T* s, T dx) {
    sFFT(S, s, dx, N, n);
}

template<typename T>
void FFT<T>::cFFT_old(T *S,T *s,T dx) {
    cFFT_old(S,s,dx,N,n,1);
}

template<typename T>
inline void FFT<T>::cadvance(T*a, int idx) {
    for (int j = 0; j <= idx - 1; j ++) {
        T a1 = a[j] + a[2 * idx - j];
        T a2 = a[j] - a[2 * idx - j];
        a[j]           = a1;
        a[2 * idx - j] = a2;
    }
}

template<typename T>
void FFT<T>::cFFT_old(T *S,T *s,T dx,int N,int n,int nr) {
    const T* ffCOS = &t.ffCOS[0];
    int sz = static_cast<int>(t.ffSIN.size());
    T*a = s;
    a[0] *= 0.5; a[N] *= 0.5; // samarskii, (15)-(16) p 66

    for (int s = 1; s <= n - 1; s++) {
        int idx = 1 << (n - s);
        int vm  = 1 << (s - 1);
        cadvance(a, idx);

        for (int k = 1; k <= idx; k++) {
            T y = 0;
            for (int j = 0; j <= idx - 1; j++) {
                y += a[idx * 2 - j] *
                    ffCOS[((2 * k - 1) * vm * nr * j) % sz];
            }
            S[(2 * k - 1) * vm] = y * dx;
        }
    }
    cadvance(a, 1 << (n-n));
    S[0]   = (a[0] + a[1]) * dx;
    S[N]   = (a[0] - a[1]) * dx;
    S[N/2] =  a[2] * dx;
}

template<typename T>
void FFT<T>::cFFT(T *S, T *s, T dx, int N, int n) {
    std::vector<T> b(N+1); // remove me
    std::vector<T> bn(N+1); // remove me
    std::vector<T>& z = b;
    std::vector<T>& zn = bn;

    T*a = s;
    a[0] *= 0.5; a[N] *= 0.5;

#define off(a,b) ((a)*(_2(m))+(b-1))
#define _off(a,b) ((a)*(_2(m-1))+(b-1))

    for (int l = n-1; l >= 1; l--) { // l=n-s
        cadvance(a, _2(l));

        int m = 0, j = 0, s = 0, k = 0;
        for (j = 0; j <= _2(l-m)-1; j++) {
            b[off(j,1)] = a[(_2(l+1))-j]; // (m=0)
        }

        // (51) p 177
        for (m = 1; m <= l-1; m++) {
            for (s = 1; s <= _2(m-1); s++) {
                j = 0;
                bn[off(j,2*s-1)] = b[_off(2*j+1,s)];
                bn[off(j,2*s)] = b[_off(2*j,s)];

                for (j = 1; j <= _2(l-m)-1; j++) {
                    bn[off(j,2*s-1)] = b[_off(2*j-1,s)]+b[_off(2*j+1,s)];
                    bn[off(j,2*s)] = b[_off(2*j,s)];
                }
            }
            bn.swap(b);
        }

        // m = l
        for (s = 1; s <= _2(m-1); s++) {
            bn[off(0,2*s-1)] = b[_off(1,s)];
            for (j = 0; j <= _2(l-m)-1; j++) {
                bn[off(j,2*s)] = b[_off(2*j,s)];
            }
        }

        bn.swap(b);

        for (s = 1; s <= _2(l); s++) {
            zn[off(1,s)] = b[off(0,s)];
        }
        zn.swap(z);

        for (m = l; m >= 1; m--) {
            for (k = 1; k <= _2(l-m); k++) {
                for (s = 1; s <= _2(m-1); s++) {
                    zn[_off(k,s)] = z[off(k,2*s)]
                        + t.iCOS(k,l-m)*z[off(k,2*s-1)];
                    zn[_off(_2(l-m+1)-k+1,s)] = z[off(k,2*s)]
                        - t.iCOS(k,l-m)*z[off(k,2*s-1)];
                }
            }
            zn.swap(z);
        }

        for (k = 1; k <= _2(l); k++) {
            S[(_2(n-l-1))*(2*k-1)] = dx*z[off(k,1)];
        }
    }

    cadvance(a, 1 << (n-n));
    S[0]   = (a[0] + a[1]) * dx;
    S[N]   = (a[0] - a[1]) * dx;
    S[N/2] =  a[2] * dx;

#undef off
#undef _off
}

template<typename T>
void FFT<T>::cFFT(T *S, T *s, T dx) {
    cFFT(S, s, dx, N, n);
}

unsigned int rev(unsigned int num, unsigned int count)
{
    unsigned int reverse_num = num;
    for (unsigned int i = 0; i < count; i++) {
        if ((num & (1 << i))) {
            reverse_num |= 1 << ((count - 1) - i);
        }
    }
    return reverse_num;
}

template<typename T>
void FFT<T>::cpFFT(T* S, T* s1, T dx) {
    vector<T> Si(N, 0);
    using cmpl = std::complex<T>;

    for (int k = 0; k < N; k++) {
        S[rev(k, n)] = s1[k];
    }

    cmpl wm,w;

    for (int s = 1; s <= n; s++) {
        const int m = _2(s);

        wm = cmpl{t.ffEXPr[s-1],t.ffEXPi[s-1]};

        for (int k = 0; k <= N-1; k += m) {
            cmpl w = 1;
            for (int j = 0; j <= m/2-1; j++) {
                cmpl t   = w * cmpl{S[k+j+m/2],Si[k+j+m/2]};
                cmpl u   =     cmpl{S[k+j    ],Si[k+j    ]};
                cmpl ut  = u+t;
                cmpl u_t = u-t;
                S [k+j]     = ut.real();
                Si[k+j]     = ut.imag();
                S [k+j+m/2] = u_t.real();
                Si[k+j+m/2] = u_t.imag();
                w = w*wm;
            }
        }
    }
}

template class FFT<double>;
template class FFT<float>;

template class FFTTable<double>;
template class FFTTable<float>;

} // namespace fdm;
