#pragma once

namespace fdm {

// Parallel Solution of Tridiagonal Linear Systems on Modern CPUs
// Tim Hennes Wichelmann
// https://www.math.uni-kiel.de/scicom/de/absolventinnen-und-absolventen/tim-wichelmann-bachelorarbeit

// Parallel Computers 2
// R W Hockney
// C R Jesshope
// p 473
template<typename T>
void cyclic_reduction(
    T *d, T *e, T *f, T *b,
    int q, int n)
{
    T alpha, gamma;
    int l, j, s, h;

    e = e-1; // e indices start from 1
    for (l = 1; l < q; l++) {
        s = 1 << l;
        h = 1 << (l-1);

        for (j = s-1; j < n-h; j += s) {
            alpha = - e[j] / d[j-h];
            gamma = - f[j] / d[j+h];
            d[j] += alpha * f[j-h] + gamma * e[j+h];
            b[j] += alpha * b[j-h] + gamma * b[j+h];
            e[j] = alpha * e[j-h];
            f[j] = gamma * f[j+h];
        }
    }

    j = (1<<(q-1)) - 1;
    b[j] = b[j] / d[j];

    for (l = q-1; l > 0; l--) {
        s = 1 << l;
        h = 1 << (l-1);
        j = h-1;

        b[j] = (b[j] - f[j] * b[j+h]) / d[j];
        for (j = h + s - 1; j < n - h; j += s) {
            b[j] = (b[j] - e[j] * b[j-h] - f[j] * b[j+h]) / d[j];
        }
        b[j] = (b[j] - e[j] * b[j-h]) / d[j];
    }
}

#ifdef __AVX2__

#include <immintrin.h>

inline void cyclic_reduction_general_avx_float(
    float *d, float *e, float *f, float *b,
    int q, int n)
{
    e = e - 1; // e indices start from 1

    for (int l = 1; l < q; l++) {
        int s = 1 << l;
        int h = 1 << (l - 1);

        for (int j = s - 1; j + h + 7 < n; j += s * 8) {
            __m256 alpha = _mm256_div_ps(
                _mm256_sub_ps(_mm256_setzero_ps(), _mm256_loadu_ps(&e[j])),
                _mm256_loadu_ps(&d[j - h])
            );
            __m256 gamma = _mm256_div_ps(
                _mm256_sub_ps(_mm256_setzero_ps(), _mm256_loadu_ps(&f[j])),
                _mm256_loadu_ps(&d[j + h])
            );

            __m256 d_vec = _mm256_loadu_ps(&d[j]);
            __m256 b_vec = _mm256_loadu_ps(&b[j]);
            __m256 e_vec = _mm256_loadu_ps(&e[j]);
            __m256 f_vec = _mm256_loadu_ps(&f[j]);

            d_vec = _mm256_add_ps(d_vec, 
                _mm256_add_ps(
                    _mm256_mul_ps(alpha, _mm256_loadu_ps(&f[j - h])),
                    _mm256_mul_ps(gamma, _mm256_loadu_ps(&e[j + h]))
                )
            );
            b_vec = _mm256_add_ps(b_vec,
                _mm256_add_ps(
                    _mm256_mul_ps(alpha, _mm256_loadu_ps(&b[j - h])),
                    _mm256_mul_ps(gamma, _mm256_loadu_ps(&b[j + h]))
                )
            );
            e_vec = _mm256_mul_ps(alpha, _mm256_loadu_ps(&e[j - h]));
            f_vec = _mm256_mul_ps(gamma, _mm256_loadu_ps(&f[j + h]));

            _mm256_storeu_ps(&d[j], d_vec);
            _mm256_storeu_ps(&b[j], b_vec);
            _mm256_storeu_ps(&e[j], e_vec);
            _mm256_storeu_ps(&f[j], f_vec);
        }

        // Handle remaining elements
        for (int j = ((n - s + 1) / s) * s - 1; j < n; j += s) {
            float alpha = -e[j] / d[j - h];
            float gamma = (j + h < n) ? -f[j] / d[j + h] : 0.0f;
            d[j] += alpha * f[j - h] + gamma * e[j + h];
            b[j] += alpha * b[j - h] + gamma * b[j + h];
            e[j] = alpha * e[j - h];
            f[j] = gamma * f[j + h];
        }
    }

    int j = std::min((1 << (q - 1)) - 1, n - 1);
    b[j] /= d[j];

    for (int l = q - 1; l > 0; l--) {
        int s = 1 << l;
        int h = 1 << (l - 1);

        for (int j = h - 1; j - h < 0; j += s) {
            b[j] = (b[j] - f[j] * b[j + h]) / d[j];
        }

        for (; j + h + 7 < n; j += s * 8) {
            __m256 b_vec = _mm256_loadu_ps(&b[j]);
            __m256 d_vec = _mm256_loadu_ps(&d[j]);
            __m256 e_vec = _mm256_loadu_ps(&e[j]);
            __m256 f_vec = _mm256_loadu_ps(&f[j]);
            __m256 b_prev = _mm256_loadu_ps(&b[j - h]);
            __m256 b_next = _mm256_loadu_ps(&b[j + h]);

            b_vec = _mm256_div_ps(
                _mm256_sub_ps(
                    b_vec,
                    _mm256_add_ps(
                        _mm256_mul_ps(e_vec, b_prev),
                        _mm256_mul_ps(f_vec, b_next)
                    )
                ),
                d_vec
            );

            _mm256_storeu_ps(&b[j], b_vec);
        }

        // Handle remaining elements
        for (; j < n; j += s) {
            b[j] = (b[j] - e[j] * b[j - h] - (j + h < n ? f[j] * b[j + h] : 0)) / d[j];
        }
    }
}

inline void cyclic_reduction_general_avx_double(
    double *d, double *e, double *f, double *b,
    int q, int n)
{
    e = e - 1; // e indices start from 1

    for (int l = 1; l < q; l++) {
        int s = 1 << l;
        int h = 1 << (l - 1);

        for (int j = s - 1; j + h + 3 < n; j += s * 4) {
            __m256d alpha = _mm256_div_pd(
                _mm256_sub_pd(_mm256_setzero_pd(), _mm256_loadu_pd(&e[j])),
                _mm256_loadu_pd(&d[j - h])
            );
            __m256d gamma = _mm256_div_pd(
                _mm256_sub_pd(_mm256_setzero_pd(), _mm256_loadu_pd(&f[j])),
                _mm256_loadu_pd(&d[j + h])
            );

            __m256d d_vec = _mm256_loadu_pd(&d[j]);
            __m256d b_vec = _mm256_loadu_pd(&b[j]);
            __m256d e_vec = _mm256_loadu_pd(&e[j]);
            __m256d f_vec = _mm256_loadu_pd(&f[j]);

            d_vec = _mm256_add_pd(d_vec, 
                _mm256_add_pd(
                    _mm256_mul_pd(alpha, _mm256_loadu_pd(&f[j - h])),
                    _mm256_mul_pd(gamma, _mm256_loadu_pd(&e[j + h]))
                )
            );
            b_vec = _mm256_add_pd(b_vec,
                _mm256_add_pd(
                    _mm256_mul_pd(alpha, _mm256_loadu_pd(&b[j - h])),
                    _mm256_mul_pd(gamma, _mm256_loadu_pd(&b[j + h]))
                )
            );
            e_vec = _mm256_mul_pd(alpha, _mm256_loadu_pd(&e[j - h]));
            f_vec = _mm256_mul_pd(gamma, _mm256_loadu_pd(&f[j + h]));

            _mm256_storeu_pd(&d[j], d_vec);
            _mm256_storeu_pd(&b[j], b_vec);
            _mm256_storeu_pd(&e[j], e_vec);
            _mm256_storeu_pd(&f[j], f_vec);
        }

        // Handle remaining elements
        for (int j = ((n - s + 1) / s) * s - 1; j < n; j += s) {
            double alpha = -e[j] / d[j - h];
            double gamma = (j + h < n) ? -f[j] / d[j + h] : 0.0;
            d[j] += alpha * f[j - h] + gamma * e[j + h];
            b[j] += alpha * b[j - h] + gamma * b[j + h];
            e[j] = alpha * e[j - h];
            f[j] = gamma * f[j + h];
        }
    }

    int j = std::min((1 << (q - 1)) - 1, n - 1);
    b[j] /= d[j];

    for (int l = q - 1; l > 0; l--) {
        int s = 1 << l;
        int h = 1 << (l - 1);

        for (int j = h - 1; j - h < 0; j += s) {
            b[j] = (b[j] - f[j] * b[j + h]) / d[j];
        }

        for (; j + h + 3 < n; j += s * 4) {
            __m256d b_vec = _mm256_loadu_pd(&b[j]);
            __m256d d_vec = _mm256_loadu_pd(&d[j]);
            __m256d e_vec = _mm256_loadu_pd(&e[j]);
            __m256d f_vec = _mm256_loadu_pd(&f[j]);
            __m256d b_prev = _mm256_loadu_pd(&b[j - h]);
            __m256d b_next = _mm256_loadu_pd(&b[j + h]);

            b_vec = _mm256_div_pd(
                _mm256_sub_pd(
                    b_vec,
                    _mm256_add_pd(
                        _mm256_mul_pd(e_vec, b_prev),
                        _mm256_mul_pd(f_vec, b_next)
                    )
                ),
                d_vec
            );

            _mm256_storeu_pd(&b[j], b_vec);
        }

        // Handle remaining elements
        for (; j < n; j += s) {
            b[j] = (b[j] - e[j] * b[j - h] - (j + h < n ? f[j] * b[j + h] : 0)) / d[j];
        }
    }
}
#endif

template<typename T>
void cyclic_reduction_general(
    T *d, T *e, T *f, T *b,
    int q, int n)
{
    if constexpr(std::is_same_v<T,float>) {
        return cyclic_reduction_general_avx_float(d, e, f, b, q, n);
    }
    if constexpr(std::is_same_v<T,double>) {
        return cyclic_reduction_general_avx_double(d, e, f, b, q, n);
    }
    T alpha, gamma;
    int l, j, s, h;

    e = e - 1; // e indices start from 1

    for (l = 1; l < q; l++) {
        s = 1 << l;
        h = 1 << (l - 1);

        for (j = s - 1; j + h < n; j += s) {
            alpha = -e[j] / d[j - h];
            gamma = -f[j] / d[j + h];
            d[j] += alpha * f[j - h] + gamma * e[j + h];
            b[j] += alpha * b[j - h] + gamma * b[j + h];
            e[j] = alpha * e[j - h];
            f[j] = gamma * f[j + h];
        }

        // Boundary, if n is not a power of 2
        for (; j < n; j += s) {
            alpha = -e[j] / d[j - h];
            d[j] += alpha * f[j - h];
            b[j] += alpha * b[j - h];
            e[j] = alpha * e[j - h];
        }
    }

    j = std::min((1 << (q - 1)) - 1, n - 1);
    b[j] /= d[j];

    for (l = q - 1; l > 0; l--) {
        s = 1 << l;
        h = 1 << (l - 1);

        for (j = h - 1; j - h < 0; j += s) {
            b[j] = (b[j] - f[j] * b[j + h]) / d[j];
        }

        for (; j + h < n; j += s) {
            b[j] = (b[j] - e[j] * b[j - h] - f[j] * b[j + h]) / d[j];
        }

        for (; j < n; j += s) {
            b[j] = (b[j] - e[j] * b[j - h]) / d[j];
        }
    }
}


} // namespace fdm
