#pragma once

extern "C" double cblas_ddot(int n, const double* x, int incx, const double* y, int incy);
extern "C" float cblas_sdot(int n, const float* x, int incx, const float* y, int incy);
extern "C" void cblas_daxpy(int n, double alpha, const double* x,int incx,double* y, int incy);
extern "C" void cblas_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy);
extern "C" double cblas_dnrm2(int n, const double* x, int incx);
extern "C" float cblas_snrm2(int n, const float* x, int incx);
extern "C" double cblas_dscal(int n, double alpha, const double* x, int incx);
extern "C" float cblas_sscal(int n, float alpha, const float* x, int incx);

extern "C" void sgtsv_(int* n, int* nrhs, float* low, float* diag, float* up, float* b, int* ldb, int*info);
extern "C" void dgtsv_(int* n, int* nrhs, double* low, double* diag, double* up, double* b, int* ldb, int*info);

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

namespace lapack {

inline void gtsv(int n, int nrhs, float* low, float* diag, float* up, float* b, int ldb, int*info)
{
    sgtsv_(&n, &nrhs, low, diag, up, b, &ldb, info);
}

inline void gtsv(int n, int nrhs, double* low, double* diag, double* up, double* b, int ldb, int*info)
{
    dgtsv_(&n, &nrhs, low, diag, up, b, &ldb, info);
}

} // namespace lapack

} // namespace fdm
