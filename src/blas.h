#pragma once

extern "C" double cblas_ddot(int n, const double* x, int incx, const double* y, int incy);
extern "C" float cblas_sdot(int n, const float* x, int incx, const float* y, int incy);
extern "C" void cblas_daxpy(int n, double alpha, const double* x,int incx,double* y, int incy);
extern "C" void cblas_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy);
extern "C" double cblas_dnrm2(int n, const double* x, int incx);
extern "C" float cblas_snrm2(int n, const float* x, int incx);
extern "C" double cblas_dscal(int n, double alpha, const double* x, int incx);
extern "C" float cblas_sscal(int n, float alpha, const float* x, int incx);

namespace fdm {
namespace blas {

// dot <- x^t y
inline double dot(int n, const double* x, int incx, const double* y, int incy) {
    return cblas_ddot(n, x, incx, y, incy);
}

inline float dot(int n, const float* x, int incx, const float* y, int incy) {
    return cblas_sdot(n, x, incx, y, incy);
}

// y <- a x + y
inline void axpy(int n, double alpha, const double* x, int incx, double* y, int incy)
{
    return cblas_daxpy(n, alpha, x, incx, y, incy);
}

inline void axpy(int n, float alpha, const float* x, int incx, float* y, int incy)
{
    return cblas_saxpy(n, alpha, x, incx, y, incy);
}

inline double nrm2(int n, const double* x, int incx) {
    return cblas_dnrm2(n, x, incx);
}

inline float nrm2(int n, const float* x, int incx) {
    return cblas_snrm2(n, x, incx);
}

inline double scal(int n, double alpha, const double* x, int incx) {
    return cblas_dscal(n, alpha, x, incx);
}

inline float scal(int n, float alpha, const float* x, int incx) {
    return cblas_sscal(n, alpha, x, incx);
}

} // namespace blas
} // namespace fdm
