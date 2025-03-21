#pragma once

#include <algorithm>
#include <iostream>
#include <vector>
#include <math.h>

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
    T *d /*middle*/, T *e /*lower, e[0] = 0*/, T *f /*upper*/, T *b,
    int q, int n)
{
    T alpha, gamma;
    int l, j, s, h;

    // std::cerr << n << " " << q << "\n";
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
    T *d /*middle*/, T *e /*lower, e[0] = 0*/, T *f /*upper*/, T *b,
    int q, int n)
{
    T alpha, gamma;
    int l, j, s, h;

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

// requires 2x memory
// effective for large n
// supports simd
template<typename T>
void cyclic_reduction_kershaw(
    T *__restrict d, T *__restrict e, T *__restrict f, T *__restrict b,
    int q, int n)
{
    T alpha, gamma;
    int j, k, l, n_curr, n_next, n_new, off, offj, offk;

    off = 0;

    for (l = 1; l < q; l++) {
        n_curr = n >> (l-1);
        n_next = n >> l;

        for (j = 0; j < n_next; j++) {
            k = 2 * j;
            offj = off + n_curr + j;
            offk = off + k;

            alpha = - e[offk+1] / d[offk];
            gamma = - f[offk+1] / d[offk+2];
            d[offj] = d[offk+1] + alpha * f[offk] + gamma * e[off + k+2];
            b[offj] = b[offk+1] + alpha * b[offk] + gamma * b[offk+2];
            e[offj] = alpha * e[offk];
            f[offj] = gamma * f[offk+2];
        }
        off += n_curr;
    }

    b[off] = b[off] / d[off];

    n_curr = 1;
    for (l = q-1; l > 0; l--) {
        n_next = n >> (l-1);
        n_new = n_next - n_curr;
        offk = off - n_next;
        b[offk] = (b[offk] - f[offk] * b[off]) / d[offk];
        b[offk + 1] = b[off];

        for (j = 1; j < n_new - 1; j++) {
            k = 2 * j;
            offj = off + j;
            offk = off - n_next + k;
            b[offk] = (b[offk] - e[offk] * b[offj - 1] - f[offk] * b[offj]) / d[offk];
            b[offk + 1] = b[offj];
        }

        j = n_new - 1;
        k = 2 * j;
        offj = off + j;
        offk = off - n_next + k;

        b[offk] = (b[offk] - e[offk] * b[offj - 1]) / d[offk];

        off -= n_next;
        n_curr = n_next;
    }
    return;
}

template<typename T>
void cyclic_reduction_kershaw_general(
    T *__restrict d, T *__restrict e, T *__restrict f, T *__restrict b,
    int q, int n)
{
    T alpha, gamma;
    int j, l, n_curr, n_next, off, dst, mask;

    off = 0;

    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;

        for (j = off + 1, dst = off + n_curr; j + 1 < off + n_curr; j += 2, dst++)
        {
            alpha = -e[j] / d[j - 1];
            gamma = -f[j] / d[j + 1];
            d[dst] = d[j] + alpha * f[j - 1] + gamma * e[j + 1];
            b[dst] = b[j] + alpha * b[j - 1] + gamma * b[j + 1];
            e[dst] = alpha * e[j - 1];
            f[dst] = gamma * f[j + 1];
        }

        // for n != 2^q-1
        for (; j < off + n_curr; j += 2, dst++) {
            alpha = -e[j] / d[j - 1];
            d[dst] = d[j] + alpha * f[j - 1];
            b[dst] = b[j] + alpha * b[j - 1];
            e[dst] = alpha * e[j - 1];
        }

        off += n_curr;
        n_curr = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1 << (q - 1)) >> 1; mask > 0; mask >>= 1) {
        if (n & mask) {
            n_next = n_curr * 2 + 1;
        } else {
            n_next = n_curr * 2;
        }

        for (j = off, dst = off-n_next + 1; j < off+n_curr; j++, dst += 2) {
            b[dst] = b[j];
        }

        j = off - n_next;
        b[j] = (b[j] - f[j] * b[j + 1]) / d[j]; j += 2;

        for (; j + 1 < off; j += 2) {
            b[j] = (b[j] - e[j] * b[j - 1] - f[j] * b[j + 1]) / d[j];
        }

        for (; j < off; j += 2) {
            b[j] = (b[j] - e[j] * b[j - 1]) / d[j];
        }

        off -= n_next;
        n_curr = n_next;
    }

    return;
}

// use precomputed d,e,f by cyclic_reduction_kershaw_general
template<typename T>
void cyclic_reduction_kershaw_general_continue(
    T *__restrict d, T *__restrict e, T *__restrict f, T *__restrict b,
    int q, int n)
{
    T alpha, gamma;
    int j, l, n_curr, n_next, off, dst, mask;

    off = 0;

    n_curr = n;

    for (l = 1; l < q; l++) {
        n_next = (n_curr - n_curr % 2) / 2;

        for (j = off + 1, dst = off + n_curr; j + 1 < off + n_curr; j += 2, dst++)
        {
            alpha = -e[j] / d[j - 1];
            gamma = -f[j] / d[j + 1];
            b[dst] = b[j] + alpha * b[j - 1] + gamma * b[j + 1];
        }

        // for n != 2^q-1
        for (; j < off + n_curr; j += 2, dst++) {
            alpha = -e[j] / d[j - 1];
            b[dst] = b[j] + alpha * b[j - 1];
        }

        off += n_curr;
        n_curr = n_next;
    }

    b[off] = b[off] / d[off];
    n_curr = 1;

    for (mask = (1 << (q - 1)) >> 1; mask > 0; mask >>= 1) {
        if (n & mask) {
            n_next = n_curr * 2 + 1;
        } else {
            n_next = n_curr * 2;
        }

        for (j = off, dst = off-n_next + 1; j < off+n_curr; j++, dst += 2) {
            b[dst] = b[j];
        }

        j = off - n_next;
        b[j] = (b[j] - f[j] * b[j + 1]) / d[j]; j += 2;

        for (; j + 1 < off; j += 2) {
            b[j] = (b[j] - e[j] * b[j - 1] - f[j] * b[j + 1]) / d[j];
        }

        for (; j < off; j += 2) {
            b[j] = (b[j] - e[j] * b[j - 1]) / d[j];
        }

        off -= n_next;
        n_curr = n_next;
    }

    return;
}

template<typename T>
class CyclicReduction {
public:
    CyclicReduction(int n)
        : n(n)
        , q(ceil(log2(n+1)))
        , Storage(4*n)
        , dUp(&Storage[0] - n)
        , eUp(&Storage[n] - n)
        , fUp(&Storage[2*n] - n)
        , bUp(&Storage[3*n] - n)
    {
    }

    void prepare(T *__restrict d, T *__restrict e, T *__restrict f)
    {
        T alpha, gamma;
        int j, l, n_curr, n_next, off, dst;

        off = 0;

        n_curr = n;

        auto loop = [&](T* d, T* e, T* f) {
            n_next = (n_curr - n_curr % 2) / 2;

            for (j = off + 1, dst = off + n_curr; j + 1 < off + n_curr; j += 2, dst++)
            {
                alpha = -e[j] / d[j - 1];
                gamma = -f[j] / d[j + 1];
                dUp[dst] = d[j] + alpha * f[j - 1] + gamma * e[j + 1];
                eUp[dst] = alpha * e[j - 1];
                fUp[dst] = gamma * f[j + 1];
            }

            // for n != 2^q-1
            for (; j < off + n_curr; j += 2, dst++) {
                alpha = -e[j] / d[j - 1];
                dUp[dst] = d[j] + alpha * f[j - 1];
                eUp[dst] = alpha * e[j - 1];
            }

            off += n_curr;
            n_curr = n_next;
        };

        l = 1;
        {
            loop(d, e, f);
        }

        for (l = 2; l < q; l++) {
            loop(&dUp[0], &eUp[0], &fUp[0]);
        }
    }

    void execute(T *__restrict d, T *__restrict e, T *__restrict f, T*__restrict b) {
        T alpha, gamma;
        int j, l, n_curr, n_next, off, dst, mask;

        off = 0;

        n_curr = n;

        auto loop1 = [&](T* d, T* e, T* f, T* b) {
            n_next = (n_curr - n_curr % 2) / 2;

            for (j = off + 1, dst = off + n_curr; j + 1 < off + n_curr; j += 2, dst++)
            {
                alpha = -e[j] / d[j - 1];
                gamma = -f[j] / d[j + 1];
                bUp[dst] = b[j] + alpha * b[j - 1] + gamma * b[j + 1];
            }

            // for n != 2^q-1
            for (; j < off + n_curr; j += 2, dst++) {
                alpha = -e[j] / d[j - 1];
                bUp[dst] = b[j] + alpha * b[j - 1];
            }

            off += n_curr;
            n_curr = n_next;
        };

        l = 1;
        {
            loop1(d, e, f, b);
        }

        for (l = 2; l < q; l++) {
            loop1(&dUp[0], &eUp[0], &fUp[0], &bUp[0]);
        }

        bUp[off] = bUp[off] / dUp[off];
        n_curr = 1;

        auto loop2 = [&](T*d, T*e, T*f, T* b, T* bDst) {
            if (n & mask) {
                n_next = n_curr * 2 + 1;
            } else {
                n_next = n_curr * 2;
            }

            for (j = off, dst = off-n_next + 1; j < off+n_curr; j++, dst += 2) {
                bDst[dst] = b[j];
            }

            j = off - n_next;
            b = bDst;
            b[j] = (b[j] - f[j] * b[j + 1]) / d[j]; j += 2;

            for (; j + 1 < off; j += 2) {
                b[j] = (b[j] - e[j] * b[j - 1] - f[j] * b[j + 1]) / d[j];
            }

            for (; j < off; j += 2) {
                b[j] = (b[j] - e[j] * b[j - 1]) / d[j];
            }

            off -= n_next;
            n_curr = n_next;
        };

        for (mask = (1 << (q - 1)) >> 1; mask > 1; mask >>= 1) {
            loop2(&dUp[0], &eUp[0], &fUp[0], &bUp[0], &bUp[0]);
        }

        mask = 1;
        loop2(d, e, f, &bUp[0], b);

        return;
    }

    void executePrepare(T *__restrict d, T *__restrict e, T *__restrict f, T*__restrict b) {
        T alpha, gamma;
        int j, l, n_curr, n_next, off, dst, mask;

        off = 0;

        n_curr = n;

        auto loop1 = [&](T* d, T* e, T* f, T* b) {
            n_next = (n_curr - n_curr % 2) / 2;

            for (j = off + 1, dst = off + n_curr; j + 1 < off + n_curr; j += 2, dst++)
            {
                alpha = -e[j] / d[j - 1];
                gamma = -f[j] / d[j + 1];
                dUp[dst] = d[j] + alpha * f[j - 1] + gamma * e[j + 1];
                eUp[dst] = alpha * e[j - 1];
                fUp[dst] = gamma * f[j + 1];
                bUp[dst] = b[j] + alpha * b[j - 1] + gamma * b[j + 1];
            }

            // for n != 2^q-1
            for (; j < off + n_curr; j += 2, dst++) {
                alpha = -e[j] / d[j - 1];
                dUp[dst] = d[j] + alpha * f[j - 1];
                eUp[dst] = alpha * e[j - 1];
                bUp[dst] = b[j] + alpha * b[j - 1];
            }

            off += n_curr;
            n_curr = n_next;
        };

        l = 1;
        {
            loop1(d, e, f, b);
        }

        for (l = 2; l < q; l++) {
            loop1(&dUp[0], &eUp[0], &fUp[0], &bUp[0]);
        }

        bUp[off] = bUp[off] / dUp[off];
        n_curr = 1;

        auto loop2 = [&](T*d, T*e, T*f, T* b, T* bDst) {
            if (n & mask) {
                n_next = n_curr * 2 + 1;
            } else {
                n_next = n_curr * 2;
            }

            for (j = off, dst = off-n_next + 1; j < off+n_curr; j++, dst += 2) {
                bDst[dst] = b[j];
            }

            j = off - n_next;
            b = bDst;
            b[j] = (b[j] - f[j] * b[j + 1]) / d[j]; j += 2;

            for (; j + 1 < off; j += 2) {
                b[j] = (b[j] - e[j] * b[j - 1] - f[j] * b[j + 1]) / d[j];
            }

            for (; j < off; j += 2) {
                b[j] = (b[j] - e[j] * b[j - 1]) / d[j];
            }

            off -= n_next;
            n_curr = n_next;
        };

        for (mask = (1 << (q - 1)) >> 1; mask > 1; mask >>= 1) {
            loop2(&dUp[0], &eUp[0], &fUp[0], &bUp[0], &bUp[0]);
        }

        mask = 1;
        loop2(d, e, f, &bUp[0], b);

        return;
    }

private:
    int n;
    int q;
    std::vector<T> Storage;
    T* dUp;
    T* eUp;
    T* fUp;
    T* bUp;
};

} // namespace fdm
