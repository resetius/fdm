#ifdef HAVE_FFTW3
#include "fft.h"
#include <cstring>

namespace fdm {

template<typename T>
void FFT_fftw3<T>::pFFT_1(T *S, T *s1, T dx)
{
    memcpy(plan.r2c_in, s1, sizeof(T) * N);
    plan.r2c_execute();

    auto* out = plan.r2c_out;
    S[0] = out[0][0] * dx;

    for (int k = 1; k < N/2; k++) {
        S[k] = out[k][0] * dx;
        S[N-k] = -out[k][1] * dx;
    }

    S[N/2] = out[N/2][0] * dx;
}

template<typename T>
void FFT_fftw3<T>::pFFT(T *S, T *s1, T dx)
{
    auto* in = plan.c2r_in;
    in[0][0] = s1[0];
    in[0][1] = 0;

    for (int k = 1; k < N/2; k++) {
        in[k][0] = s1[k];
        in[k][1] = -s1[N-k];
    }
    in[N/2][0] = s1[N/2];
    in[N/2][1] = 0;

    plan.c2r_execute();
    for (int i = 0; i < N; i++) {
        S[i] = plan.c2r_out[i] * dx * 0.5;
    }
}

template<typename T>
void FFT_fftw3<T>::sFFT(T *S, T *s, T dx)
{
    memcpy(plan.dst1_in, s+1, sizeof(T) * (N-1));
    plan.dst1_execute();

    for (int k = 1; k < N; k++) {
        S[k] = plan.dst1_out[k-1] * dx * 0.5;
    }
}

template<typename T>
void FFT_fftw3<T>::cFFT(T *S, T *s, T dx)
{
    memcpy(plan.dct1_in, s, sizeof(T) * (N+1));
    plan.dct1_execute();

    for (int k = 0; k <= N; k++) {
        S[k] = plan.dct1_out[k] * dx * 0.5;
    }
}

template class FFT_fftw3<double>;
template class FFT_fftw3<float>;

} // namespace fdm
#endif // HAVE_FFTW3
