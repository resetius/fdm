#include <cmath>
#include <string.h>

#include "fft.h"
#include "verify.h"

using namespace std;

namespace fdm {

template<typename T>
void FFTTable<T>::init(int N) {
    ffCOS.resize(2*N);
    ffSIN.resize(2*N);

    for (int m = 0; m < 2*N; ++m) {
        ffCOS[m] = cos(m * M_PI/N);
        ffSIN[m] = sin(m * M_PI/N);
    }
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
void FFT<T>::pFFT_1(T *S, T *s1, T dx) {
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
void FFT<T>::pFFT(T *S, T* s, T dx) {
    int N_2 = N/2;
    int k;

    cFFT(&S[0], &s[0], dx, N_2,n-1,2);

    // S[N_2] not filled, N_2+1 first non empty
    sFFT(&s[0], &s[N_2], dx, N_2,n-1,2);

    for (k = 1; k <= N_2 - 1; k ++) {
        int r = k%2;
        T S_k   = (S[k] + (2*r-1)*s[k]);
        T S_N_k = (S[k] - (2*r-1)*s[k]);
        S[k]    = S_k;
        S[N-k]  = S_N_k;
    }
}

template<typename T>
void FFT<T>::sFFT(T *S,T *s,T dx) {
    sFFT(S, s, dx, N, n, 1);
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
void FFT<T>::sFFT(T *S,T *s,T dx,int N,int n,int nr) {
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
void FFT<T>::sFFT2(T* S, T* s, T dx) {
    std::vector<T> b(N); // remove me
    std::vector<T> z(N); // remove me

    T* a = s;
#define off(a,b) 0
    for (int l = n-1; l >= 1; l--) { // l=n-s
        // (30), p 170
        sadvance(a, 1<<l); // a^s

        // (36), p 172
        // b^0 = a
        int m = 0, j = 0, s = 0, k = 0;
        for (j = 1; j <= 1<<(l-m); j++) {
            b[off(j,1)] = a[(1<<(l+1))-j]; // (m=0)
        }
        // b^m, m = 1, ... l, incr
        for (m = 1; m <= l-1; m++) {
            for (s = 1; s <= 1<<(m-1); s++) {
                for (j = 1; j <= (1<<(l-m))-1; j++) {
                    b[off(j,2*s-1)] = b[off(2*j-1,s)]+b[off(2*j+1,s)];
                    b[off(j,2*s)]   = b[off(2*j,s)];
                }
                b[off(j,2*s)] = b[off(2*j,s)];
            }
        }
        for (s = 1; s <= 1<<(m-1); s++) {
            b[off(1<<(l-m),2*s-1)] = b[off(1<<(l-m+1)-1,s)];
            b[off(1<<(l-m)-1,2*s)] = b[off(1<<(l-m+1)-2,s)];
            b[off(1<<(l-m),2*s)]   = b[off(1<<(l-m+1),s)];
        }

        // (37), p 172
        // z^l = b^l
        for (s = 1; s <= 1<<l; s++) {
            z[off(1,s)] = b[off(1,s)];
        }
        // z^m, m = l, ...,0, decr
        for (m = l; m >= 1; m--) {
            for (k = 1; k <= 1<<(l-m); k++) {
                z[off(k,s)] = z[off(k,2*s)]
                    + 1.0/(2.0*cos(M_PI*(2*k-1)/(1<<(l-m+2))))*z[off(k,2*s-1)];
                z[off(1<<(l-m+1)-k+1,s)] = -z[off(k,2*2)]
                    + 1.0/(2.0*cos(M_PI*(2*k-1)/(1<<(l-m+2))))*z[off(k,2*s-1)];
            }
        }
        // z^0 -> y (ans)
        for (k = 1; k <= 1<<(l-m); k++) {
            S[1<<(s-1)*(2*k-1)] = z[off(k,1)];
        }
    }
#undef off
}

template<typename T>
void FFT<T>::cFFT(T *S,T *s,T dx) {
    cFFT(S,s,dx,N,n,1);
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
void FFT<T>::cFFT(T *S,T *s,T dx,int N,int n,int nr) {
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

template class FFT<double>;
template class FFT<float>;

template class FFTTable<double>;
template class FFTTable<float>;

} // namespace fdm;
