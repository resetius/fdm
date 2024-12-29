#ifdef HAVE_FFTW3
#include "fft.h"

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

}

template class FFT_fftw3<double>;
template class FFT_fftw3<float>;

} // namespace fdm
#endif // HAVE_FFTW3