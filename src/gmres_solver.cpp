#include "gmres_solver.h"
#include "blas.h"

namespace fdm {

using namespace fdm;

template<typename T>
void gmres_solver<T>::solve(T* x, T* b)
{
    T e = 1.0;
    T bn  = blas::nrm2(n, b, 1);
    memcpy(x, b, n*sizeof(T));
    for (int i = 0; i < maxit; i++) {
        e = algorithm6_9(x, &b[0], tol * bn);
        e /= bn;
        if (e < tol) {
            break;
        }
    }
}

/**
 * Demmel Algorithm  6.9 p 303
 */
template<typename T>
T gmres_solver<T>::algorithm6_9(T * x, const T* b, T eps)
{
    r.resize(n); /* b - Ax */
    ax.resize(n);

    T gamma_0;
    T beta;
    T ret;

    int i, j;

    int hz = k + 1;

    /* x0 = x */
    /* r0 = b - Ax0 */

    mat.mul(&ax[0], x);
    memcpy(&r[0], b, n*sizeof(T));
    axpy(n, -1, &ax[0], 1, &r[0], 1);

    gamma_0 = nrm2(n, &r[0], 1);

    if (gamma_0 <= eps) {
        ret = gamma_0;
        return ret;
    }

    h.resize(hz * hz); memset(&h[0], 0, h.size()*sizeof(T));
    q.resize(hz * n); memcpy(&q[0], &r[0], n*sizeof(T));
    scal(n, 1.0 / gamma_0, &q[0], 1);
    gamma.resize(hz);
    s.resize(hz); memset(&s[0], 0, hz*sizeof(T));
    c.resize(hz); memset(&c[0], 0, hz*sizeof(T));

    gamma[0] = gamma_0;

    for (j = 0; j < k; ++j) {
        double nr1, nr2;

        mat.mul(&ax[0], &q[j * n]);
        nr1 = nrm2(n, &ax[0], 1);

        for (i = 0; i <= j; ++i) {
            h[i * hz + j] = dot(n, &q[i * n], 1, &ax[0], 1); //-> j x j
            //ax = ax - h[i * hz + j] * q[i * n];
            axpy(n, -h[i * hz + j], &q[i * n], 1, &ax[0], 1);
        }

        // h -> (j + 1) x j
        h[(j + 1) * hz + j] = nrm2(n, &ax[0], 1);

        // loss of orthogonality detected
        // C. T. Kelley: Iterative Methods for Linear and Nonlinear Equations, SIAM, ISBN 0-89871-352-8
        nr2 = 0.001 * h[(j + 1) * hz + j] + nr1;
        if (fabs(nr2  - nr1) < eps) {
            /*fprintf(stderr, "reortho!\n");*/
            for (i = 0; i <= j; ++i) {
                T hr = dot(n, &q[i * n], 1, &ax[0], 1);
                h[i * hz + j] += hr;
                axpy(n, -hr, &q[i * n], 1, &ax[0], 1);
            }
            h[(j + 1) * hz + j] = nrm2(n, &ax[0], 1);
        }

        // rotate
        for (i = 0; i <= j - 1; ++i) {
            T x = h[i * hz + j];
            T y = h[(i + 1) * hz + j];

            h[i * hz + j]       = x * c[i + 1] + y * s[i + 1];
            h[(i + 1) * hz + j] = x * s[i + 1] - y * c[i + 1];
        }

        beta = std::sqrt(h[j * hz + j] * h[j * hz + j] + h[(j + 1) * hz + j] * h[(j + 1) * hz + j]);
        s[j + 1]      = h[(j + 1) * hz + j] / beta;
        c[j + 1]      = h[j * hz + j] / beta;
        h[j * hz + j] = beta;

        gamma[j + 1] = s[j + 1] * gamma[j];
        gamma[j]     = c[j + 1] * gamma[j];

        if (gamma[j + 1]  > eps) {
            memset(&q[(j + 1) * n], 0, n*sizeof(T));
            axpy(n, 1.0 / h[(j + 1) * hz + j], &ax[0], 1, &q[(j + 1) * n], 1);
        } else {
            goto done;
        }
    }

    --j;

done:
    ret = gamma[j + 1];

    {
        y.resize(hz);
        for (i = j; i >= 0; --i) {
            double sum = 0.0;
            for (k = i + 1; k <= j; ++k) {
                sum += h[i * hz + k] * y[k];
            }
            y[i] = (gamma[i] - sum) / h[i * hz + i];
        }

        for (i = 0; i <= j; ++i) {
            axpy(n, y[i], &q[i * n], 1, x, 1);
        }
    }

    return ret;
}

template class gmres_solver<double>;
template class gmres_solver<float>;

} // namespace fdm
