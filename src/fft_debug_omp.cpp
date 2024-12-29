#include "fft.h"
#include "verify.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace std::chrono;

namespace fdm {

#define _2(a) (1<<(a))

namespace {

#ifdef _OPENMP
template<typename T>
inline void padvance_omp(T* a, int idx, int id, int work)
{
    for (int j = id; j<id+work&&j <= idx - 1; j++) {
        T a1, a2;
        a1 = a[j] + a[idx + j];
        a2 = a[j] - a[idx + j];
        a[j]       = a1;
        a[idx + j] = a2;
    }
}

template<typename T>
inline void sadvance_omp(T* a, int idx, int id, int work) {
    for (int j = id+1; j < id+1+work && j <= idx-1; j++) {
        T a1 = a[j] - a[2 * idx - j];
        T a2 = a[j] + a[2 * idx - j];
        a[j]           = a1;
        a[2 * idx - j] = a2;
    }
}

template<typename T>
inline void cadvance_omp(T*a, int idx, int id, int work) {
    for (int j = id; j < id+work && j <= idx - 1; j ++) {
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

template<typename T>
inline void cadvance(T*a, int idx) {
    for (int j = 0; j <= idx - 1; j ++) {
        T a1 = a[j] + a[2 * idx - j];
        T a2 = a[j] - a[2 * idx - j];
        a[j]           = a1;
        a[2 * idx - j] = a2;
    }
}

#endif

} // namespace {

template<typename T>
void FFT_debug_omp<T>::init() {
    int n1 = N;
    n = 0;
    while (n1 % 2 == 0) {
        n1 /= 2; n++;
    }

    verify((1<<n) == N);
}

template<typename T>
void FFT_debug_omp<T>::pFFT_1(T *S, T *s1, T dx) {
#ifdef _OPENMP
    std::vector<T> b(2*N); // remove me

    int N_2 = N/2;
    int yoff = 0;
    int _yoff = N_2;

    T* a = s1;

#define off(a,b) ((a)*(_2(m))+(b-1))
#define _off(a,b) ((a)*(_2(m-1))+(b-1))

#define zoff(a,b) ((a-1)*(_2(m))+(b-1))
#define _zoff(a,b) ((a-1)*(_2(m-1))+(b-1))

    int size = 4; // _2(n-1);

    omp_set_dynamic(0);
    omp_set_num_threads(size);

#pragma omp parallel
    {
    int thread_id = omp_get_thread_num();
    int work = std::max(2, _2(n) / size);
    int boff = 0;
    int nboff = boff+N;

    for (int l = n-1; l >= 2; l--) { // l=n-s
        work = max(1, _2(l)/size);
        int id = thread_id*work;

        padvance_omp(a, _2(l), id, work);

#pragma omp barrier
        int m = 0, j = 0, s = 0, k = 0, i = 0;
        for (j = id; j<id+work&&j <= _2(l)-1; j++) {
            b[boff+off(j,1)] = a[_2(l)+j];
        }

        for (m = 1; m <= l-1; m++) {
            int ns = _2(m-1);
            int nj = _2(l-m)-1;

#pragma omp barrier
            for (int i = id; i < id+work; i++) {
                j = i/ns;
                s = i%ns+1;

                if (j == 0) {
                    b[nboff+off(j,2*s-1)] = b[boff+_off(1,s)]-b[boff+_off(_2(l-m+1)-1,s)];
                    b[nboff+off(j,2*s)] = b[boff+_off(2*j,s)];
                } else if (j <= nj) {
                    b[nboff+off(j,2*s-1)] = b[boff+_off(2*j-1,s)] + b[boff+_off(2*j+1,s)];
                    b[nboff+off(j,2*s)] = b[boff+_off(2*j,s)];
                }
            }
            swap(nboff, boff);
        }

#pragma omp barrier
        work = max(1, _2(l-1)/size);
        id = thread_id*work;
        m=l-1;
        for (s = id+1; s<id+1+work&&s <= _2(l-1); s++) {
            b[nboff+zoff(1,s)] = b[boff+off(0,s)];
            b[nboff+_yoff+zoff(1,s)] = b[boff+off(1,s)];
        }
        swap(nboff, boff);

        for (m = l-1; m >= 1; m--) {
            int ns = _2(m-1);
            int nk = _2(l-m-1);
#pragma omp barrier
            for (i = id; i <id+work; i++) {
                s=i%ns+1;
                k=i/ns+1;
                if (k <= nk) {
                    b[nboff+_zoff(k,s)] = b[boff+zoff(k,2*s)]
                        + t.iCOS(k,l-m-1)*b[boff+zoff(k,2*s-1)];
                    b[nboff+_zoff(_2(l-m)-k+1,s)] = b[boff+zoff(k,2*s)]
                        - t.iCOS(k,l-m-1)*b[boff+zoff(k,2*s-1)];
                    b[nboff+_yoff+_zoff(k,s)] = b[boff+_yoff+zoff(k,2*s)]
                        + t.iCOS(k,l-m-1)*b[boff+_yoff+zoff(k,2*s-1)];
                    b[nboff+_yoff+_zoff(_2(l-m)-k+1,s)] = -b[boff+_yoff+zoff(k,2*s)]
                        + t.iCOS(k,l-m-1)*b[boff+_yoff+zoff(k,2*s-1)];
                }
            }
            swap(nboff, boff);
        }

#pragma omp barrier
        for (k = id+1; k<id+1+work&&k <= _2(l-1); k++) {
            S[yoff+_2(n-l-1)*(2*k-1)] = dx*b[boff+zoff(k,1)];
            S[N-(_2(n-l-1)*(2*k-1))] = dx*b[boff+_yoff+zoff(k,1)];
        }
    }
    }

    padvance(a, 1 << (n - (n-1)));
    S[yoff +(1 << (n - 2))] = dx*a[2];
    S[N-(1 << (n - 2))] = dx*a[3];

    padvance(a, 1 << (n - n));
    S[yoff + 0]             = dx*a[0];
    S[yoff + N_2]           = dx*a[1];

#undef off
#undef _off
#undef zoff
#undef _zoff

#else // #ifdef _OPENMP
    abort();
#endif
}

template<typename T>
void FFT_debug_omp<T>::sFFT(T* S, T* s, T dx) {
#ifdef _OPENMP
    std::vector<T> b(2*N); // remove me

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
    int work = std::max(2, _2(n) / size);
    int boff = 0;
    int nboff = boff+N/2;

    time_point<steady_clock> t1;
    if (thread_id == 0) t1 = steady_clock::now();
    for (int l = n-1; l >= 1; l--) { // l=n-s
        work = max(1, work>>1);
        int id = thread_id*work;

        sadvance_omp(a, _2(l), id, work);

#pragma omp barrier
        int m = 0, j = 0, s = 0, k = 0, i = 0;
        for (j = id+1; j < id+1+work && j <= _2(l); j++) {
            b[boff+off(j,1)] = a[(_2(l+1))-j];
        }

        for (m = 1; m <= l-1; m++) {
            int ns = _2(m-1);
            int nj = _2(l-m);
#pragma omp barrier
            for (int i = id; i < id+work; i++) {
                j = i/ns + 1;
                s = i%ns + 1;
                if (j <= nj) {
                    if (j == nj) {
                        b[nboff+off(j,2*s-1)] = b[boff+_off(2*j-1,s)];
                    } else {
                        b[nboff+off(j,2*s-1)] = b[boff+_off(2*j-1,s)]+b[boff+_off(2*j+1,s)];
                    }
                    b[nboff+off(j,2*s)]   = b[boff+_off(2*j,s)];
                }
            }
            swap(boff, nboff);
        }

        int ns = _2(m-1);
#pragma omp barrier
        for (s = id + 1; s < id+1+work && s <= ns; s++) {
            b[nboff+off(1,2*s)] = b[boff+_off(2,s)];
            b[nboff+off(1,2*s-1)] = b[boff+_off(1,s)];
        }
        swap(boff, nboff);

        for (m = l; m >= 1; m--) {
            int ns = _2(m-1);
            int nk = _2(l-m);
#pragma omp barrier
            for (i = id, k=i/ns+1, s=i%ns+1;
                 k <= nk && i < id+work;
                 i++, k=i/ns+1,s=i%ns+1)
            {
                b[nboff+_off(k,s)] = b[boff+off(k,2*s)]
                    + t.iCOS(k,l-m)*b[boff+off(k,2*s-1)];
                b[nboff+_off(_2(l-m+1)-k+1,s)] = -b[boff+off(k,2*s)]
                    + t.iCOS(k,l-m)*b[boff+off(k,2*s-1)];
            }
            swap(boff, nboff);
        }

#pragma omp barrier
        for (k = id+1; k < id+1+work&&k <= _2(l); k++) {
            S[(_2(n-l-1))*(2*k-1)] = dx*b[boff+off(k,1)];
        }
    }
    if (thread_id == 0) {
        auto t2 = steady_clock::now();
        auto interval = duration_cast<duration<double>>(t2 - t1);
        //printf("time=%f\n", interval.count());
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
void FFT_debug_omp<T>::cFFT(T *S, T *s, T dx) {
#ifdef _OPENMP
    std::vector<T> b(2*N); // remove me

    T*a = s;
    a[0] *= 0.5; a[N] *= 0.5;

#define off(a,b) ((a)*(_2(m))+(b-1))
#define _off(a,b) ((a)*(_2(m-1))+(b-1))

    int size = 4; // _2(n-1);

    omp_set_dynamic(0);
    omp_set_num_threads(size);

#pragma omp parallel
    {
    int thread_id = omp_get_thread_num();
    int work = std::max(2, _2(n) / size);
    int boff = 0;
    int nboff = boff+N;

    for (int l = n-1; l >= 1; l--) { // l=n-s
        work = max(1, work>>1);
        int id = thread_id*work;

        cadvance_omp(a, _2(l), id, work);

#pragma omp barrier
        int m = 0, j = 0, s = 0, k = 0, i = 0;
        for (j = id; j <id+work && j <= _2(l)-1; j++) {
            b[boff+off(j,1)] = a[(_2(l+1))-j]; // (m=0)
        }

        // (51) p 177
        for (m = 1; m <= l-1; m++) {
            int ns = _2(m-1);
            int nj = _2(l-m)-1;

#pragma omp barrier
            for (int i = id; i < id+work; i++) {
                j = i/ns;
                s = i%ns+1;
                if (j <= nj) {
                    if (j == 0) {
                        b[nboff+off(j,2*s-1)] = b[boff+_off(2*j+1,s)];
                        b[nboff+off(j,2*s)] = b[boff+_off(2*j,s)];
                    } else {
                        b[nboff+off(j,2*s-1)] = b[boff+_off(2*j-1,s)]+b[boff+_off(2*j+1,s)];
                        b[nboff+off(j,2*s)] = b[boff+_off(2*j,s)];
                    }
                }
            }
            swap(nboff, boff);
        }

        // m = l
#pragma omp barrier
        for (s = id+1; s<id+1+work && s <= _2(m-1); s++) {
            b[nboff+off(0,2*s-1)] = b[boff+_off(1,s)];
            b[nboff+off(0,2*s)] = b[boff+_off(0,s)];
        }
        swap(nboff, boff);

#pragma omp barrier
        for (s = id+1; s<id+1+work && s <= _2(l); s++) {
            b[nboff+off(1,s)] = b[boff+off(0,s)];
        }
        swap(nboff, boff);

        for (m = l; m >= 1; m--) {
            int ns = _2(m-1);
            int nk = _2(l-m);

#pragma omp barrier
            for (i = id; i < id+work; i++) {
                k=i/ns+1;
                s=i%ns+1;
                if (k <= nk) {
                    b[nboff+_off(k,s)] = b[boff+off(k,2*s)]
                        + t.iCOS(k,l-m)*b[boff+off(k,2*s-1)];
                    b[nboff+_off(_2(l-m+1)-k+1,s)] = b[boff+off(k,2*s)]
                        - t.iCOS(k,l-m)*b[boff+off(k,2*s-1)];
                }
            }
            swap(nboff, boff);
        }

#pragma omp barrier
        for (k = id+1; k<id+1+work && k <= _2(l); k++) {
            S[(_2(n-l-1))*(2*k-1)] = dx*b[boff+off(k,1)];
        }
    }
    } // omp

    cadvance(a, 1 << (n-n));
    S[0]   = (a[0] + a[1]) * dx;
    S[N]   = (a[0] - a[1]) * dx;
    S[N/2] =  a[2] * dx;

#undef off
#undef _off

#else // #ifdef _OPENMP
    abort();
#endif
}

template class FFT_debug_omp<double>;
template class FFT_debug_omp<float>;

} // namespace fdm
