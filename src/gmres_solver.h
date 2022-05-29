#pragma once

#include "sparse.h"

namespace fdm {

template<typename T>
class gmres_solver {
    int k;
    int maxit;
    int n = 0;
    T tol;
    csr_matrix<T> mat;

    std::vector<T> r,ax,h,q,gamma,s,c,y;

public:
    gmres_solver(int kdim=1000, int maxit=100, double tol=1e-6)
        : k(kdim)
        , maxit(maxit)
        , tol(tol)
    {
    }

    gmres_solver& operator=(csr_matrix<T>&& matrix)
    {
        mat = std::move(matrix);
        n = static_cast<int>(mat.Ap.size()) - 1;
        return *this;
    }

    void solve(T* x, T* b)
    {
        T e = 1.0;
        T bn  = vec_norm2(b, n);
        memcpy(x, b, n*sizeof(T));
        for (int i = 0; i < maxit; i++) {
            e = algorithm6_9(x, &b[0], tol * bn);
            e /= bn;
            if (e < tol) {
                break;
            }
        }
    }

private:
    static void vec_diff(T* r, const T* a, const T* b, int n)
    {
        for (int i = 0; i < n; ++i) {
            r[i] = a[i] - b[i];
        }
    }

    static void vec_mult_scalar(T* a, const T* b, T k, int n)
    {
        for (int i = 0; i < n; ++i) {
            a[i] = b[i] * k;
        }
    }

    static void vec_sum2(T* r, const T* a, const T* b, T k2, int n)
    {
        for (int i = 0; i < n; ++i) {
            r[i] = a[i] + b[i] * k2;
        }
    }

    static T vec_scalar2(const T* a, const T* b, int n)
    {
        T s = 0;
        for (int i = 0; i < n; ++i) {
            s += a[i] * b[i];
        }
        return s;
    }

    static T vec_norm2(const T* v, int n)
    {
        return std::sqrt(vec_scalar2(v, v, n));
    }

    /**
     * Demmel Algorithm  6.9 p 303
     */
    T algorithm6_9(T * x, const T* b, T eps)
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
        vec_diff(&r[0], b, &ax[0], n);

        gamma_0 = vec_norm2(&r[0], n);

        if (gamma_0 <= eps) {
            ret = gamma_0;
            return ret;
        }

        h.resize(hz * hz); memset(&h[0], 0, h.size()*sizeof(T));
        q.resize(hz * n);
        vec_mult_scalar(&q[0], &r[0], 1.0 / gamma_0, n);
        gamma.resize(hz);
        s.resize(hz); memset(&s[0], 0, hz*sizeof(T));
        c.resize(hz); memset(&c[0], 0, hz*sizeof(T));

        gamma[0] = gamma_0;

        for (j = 0; j < k; ++j) {
            double nr1, nr2;

            mat.mul(&ax[0], &q[j * n]);
            nr1 = vec_norm2(&ax[0], n);

            for (i = 0; i <= j; ++i) {
                h[i * hz + j] = vec_scalar2(&q[i * n], &ax[0], n); //-> j x j
                //ax = ax - h[i * hz + j] * q[i * n];
                vec_sum2(&ax[0], &ax[0], &q[i * n], -h[i * hz + j], n);
            }

            // h -> (j + 1) x j
            h[(j + 1) * hz + j] = vec_norm2(&ax[0], n);

            // loss of orthogonality detected
            // C. T. Kelley: Iterative Methods for Linear and Nonlinear Equations, SIAM, ISBN 0-89871-352-8
            nr2 = 0.001 * h[(j + 1) * hz + j] + nr1;
            if (fabs(nr2  - nr1) < eps) {
                /*fprintf(stderr, "reortho!\n");*/
                for (i = 0; i <= j; ++i) {
                    T hr = vec_scalar2(&q[i * n], &ax[0], n);
                    h[i * hz + j] += hr;
                    vec_sum2(&ax[0], &ax[0], &q[i * n], -hr, n);
                }
                h[(j + 1) * hz + j] = vec_norm2(&ax[0], n);
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
                vec_mult_scalar(&q[(j + 1) * n], &ax[0], 1.0 / h[(j + 1) * hz + j], n);
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
                vec_sum2(x, x, &q[i * n], y[i], n);
            }
        }

        return ret;
    }
};

} // namespace fdm
