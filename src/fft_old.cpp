#include "fft.h"
#include "verify.h"

namespace fdm {

namespace {

template<typename T>
inline void sadvance(T*a, int idx) {
    for (int j = 1; j <= idx - 1; j ++) {
        T a1 = a[j] - a[2 * idx - j];
        T a2 = a[j] + a[2 * idx - j];
        a[j]           = a1;
        a[2 * idx - j] = a2;
    }
}

template<typename T>
inline void cadvance(T*a, int idx) {
    for (int j = 0; j <= idx - 1; j ++) {
        T a1 = a[j] + a[2 * idx - j];
        T a2 = a[j] - a[2 * idx - j];
        a[j]           = a1;
        a[2 * idx - j] = a2;
    }
}

template<typename T>
inline void padvance(T* a, int idx)
{
    for (int j = 0; j <= idx - 1; j++) {
        T a1, a2;
        a1 = a[j] + a[idx + j];
        a2 = a[j] - a[idx + j];
        a[j]         = a1;
        a[idx + j] = a2;
    }
}

} // namespace {

template<typename T>
void FFT_old<T>::init() {
    int n1 = N;
    n = 0;
    while (n1 % 2 == 0) {
        n1 /= 2; n++;
    }

    verify((1<<n) == N);
}

template<typename T>
void FFT_old<T>::pFFT_1(T *S, T *s1, T dx) {
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
void FFT_old<T>::sFFT(T *S,T *s,T dx) {
    sFFT(S, s, dx, N, n, 1);
}

template<typename T>
void FFT_old<T>::sFFT(T *S,T *s,T dx,int N,int n,int nr) {
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
void FFT_old<T>::cFFT(T *S,T *s,T dx) {
    cFFT(S,s,dx,N,n,1);
}

template<typename T>
void FFT_old<T>::cFFT(T *S,T *s,T dx,int N,int n,int nr) {
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

template class FFT_old<double>;
template class FFT_old<float>;

} // namespace fdm
