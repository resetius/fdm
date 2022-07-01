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

    a.resize((n+1)*N);
    y.resize((N/2+1));
    _y.resize((N/2+1));
    y1.resize((N/2+1));
    y2.resize((N/2+1));
    ss.resize((N/2+1));
}

template<typename T>
void FFT<T>::pFFT_1(T *S, const T *s1, T dx) {
    const T* ffSIN = &t.ffSIN[0];
    const T* ffCOS = &t.ffCOS[0];

    int p, s, j, idx, idx2, vm, k;
    int sz = static_cast<int>(t.ffSIN.size());
    int N_2 = N/2;
    int yoff = 0;
    int _yoff = N_2;

    memcpy(&a[0], s1, N * sizeof(T));

    for (p = 1; p <= n; p++) {
        idx = 1 << (n - p);
        for (j = 0; j <= idx - 1; j++) {
            a[p * N + j]       = a[(p - 1) * N + j] + a[(p - 1) * N + idx + j];
            a[p * N + idx + j] = a[(p - 1) * N + j] - a[(p - 1) * N + idx + j];
        }
    }

    for (s = 1; s <= n - 2; s++) {
        idx = 1 << (n - s - 1);
        vm  = 1 << s;
        for (k = 1; k <= idx; k++) {
            T s1 = 0.0;
            T s2 = 0.0;
            for (j = 0; j <= 2 * idx - 1; j++) {
                s1 += a[s * N + 2 * idx + j] *
                    ffCOS[((2 * k - 1) * vm * j) % sz];
            }
            for (j = 1; j <= 2 * idx - 1; j++) {
                s2 += a[s * N + 2 * idx + j] *
                    ffSIN[((2 * k - 1) * vm * j) % sz];
            }
            idx2 = (1 << (s - 1)) * (2 * k - 1);
            S[yoff + idx2]  = s1;
            S[_yoff + idx2] = s2;
        }
    }

    S[yoff + N_2]           = a[n * N + 1];
    S[yoff+(1 << (n - 2))]  = a[(n - 1) * N + 2];

    S[yoff + 0]             = a[n * N + 0];
    S[_yoff+(1 << (n - 2))] = a[(n - 1) * N + 3];

    for (k = 0; k <= N_2; k++) {
        S[yoff + k]  = S[yoff + k] * dx;
    }

    for (int k = 1; k <= N_2-1; k++) {
        S[_yoff + k] = S[_yoff + k] * dx;
    }
    for (int k = 1; k < N_2/2; k++) {
        T tmp = S[_yoff + k];
        S[_yoff + k] = S[_yoff+N_2-k];
        S[_yoff+N_2-k] = tmp;
    }
}

template<typename T>
void FFT<T>::pFFT(T *S, const T* s, T dx) {
    int N_2 = N/2;
    int k;

    memcpy(&ss[0], &s[0], (N_2+1) * sizeof(T));

    cFFT(&y1[0], &ss[0], dx, N_2,n-1,2);

    for (k = 1; k <= N_2-1; k++) {
        ss[k] = s[N-k];
    }
    sFFT(&y2[0], &ss[0], dx, N_2,n-1,2);

    for (k = 1; k <= N_2 - 1; k ++) {
        S[k]   = (y1[k] + y2[k]);
        S[N-k] = (y1[k] - y2[k]);
    }
    S[0]   = y1[0]  ;
    S[N_2] = y1[N_2];
}

template<typename T>
void FFT<T>::sFFT(T *S,T *s,T dx) {
    sFFT(S, s, dx, N, n, 1);
}

template<typename T>
void FFT<T>::sFFT(T *S,T *s,T dx,int N,int n,int nr) {
    const T* ffSIN = &t.ffSIN[0];
    int sz = static_cast<int>(t.ffSIN.size());

    memcpy(&a[1], &s[1], (N - 1) * sizeof(T));

    for (int p = 1; p <= n - 1; p++) {
        int idx = 1 << (n - p);
        for (int j = 1; j <= idx - 1; j ++) {
            a[p * N + j]           = a[(p - 1) * N + j] - a[(p - 1) * N + 2 * idx - j];
            a[p * N + 2 * idx - j] = a[(p - 1) * N + j] + a[(p - 1) * N + 2 * idx - j];
            a[p * N + idx]         = a[(p - 1) * N + idx];
        }
    }

    for (int s = 1; s <= n - 1; s++) {
        int idx = 1 << (n - s);
        int vm  = 1 << (s - 1);
        for (int k = 1; k <= idx; k++) {
            T y = 0;
            for (int j = 1; j <= idx; j++) {
                y += a[s * N + idx * 2 - j] *
                    ffSIN[((2 * k - 1) * vm * nr * j) % sz];

            }
            S[(2 * k - 1) * vm] = y * dx;
        }
    }
    int idx = 1 << (n - 1);
    S[idx] = a[(n - 1) * N + 1] * dx;
}

template<typename T>
void FFT<T>::cFFT(T *S,T *s,T dx) {
    cFFT(S,s,dx,N,n,1);
}

template<typename T>
void FFT<T>::cFFT(T *S,T *s,T dx,int N,int n,int nr) {
    const T* ffCOS = &t.ffCOS[0];
    int sz = static_cast<int>(t.ffSIN.size());
    int M = N + 1;
    memcpy(&a[0], &s[0], M * sizeof(T));
    a[0] *= 0.5; a[N] *= 0.5; // samarskii, (15)-(16) p 66

    for (int p = 1; p <= n; p++) {
        int idx = 1 << (n - p);
        for (int j = 0; j <= idx - 1; j ++) {
            a[p * M + j]           = a[(p - 1) * M + j] + a[(p - 1) * M + 2 * idx - j];
            a[p * M + 2 * idx - j] = a[(p - 1) * M + j] - a[(p - 1) * M + 2 * idx - j];
            a[p * M + idx]         = a[(p - 1) * M + idx];
        }
    }

    for (int s = 1; s <= n - 1; s++) {
        int idx = 1 << (n - s);
        int vm  = 1 << (s - 1);
        for (int k = 1; k <= idx; k++) {
            T y = 0;
            for (int j = 0; j <= idx - 1; j++) {
                y += a[s * M + idx * 2 - j] *
                    ffCOS[((2 * k - 1) * vm * nr * j) % sz];
            }
            S[(2 * k - 1) * vm] = y * dx;
        }
    }
    S[0]   = (a[n * M + 0] + a[n * M + 1]) * dx;
    S[N]   = (a[n * M + 0] - a[n * M + 1]) * dx;
    S[N/2] =  a[n * M + 2] * dx;
}

template class FFT<double>;
template class FFT<float>;

template class FFTTable<double>;
template class FFTTable<float>;

} // namespace fdm;
