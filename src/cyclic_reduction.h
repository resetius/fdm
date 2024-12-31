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

template<typename T>
void cyclic_reduction_general(
    T *d, T *e, T *f, T *b,
    int q, int n)
{
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