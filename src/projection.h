#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "blas.h"

namespace fdm {

/**
   Скалярное произведение по умолчанию. Функтор, а не указатель на функцию:
   неортогональной проекции нужно уметь принимать и произведение с весом,
   например \f$(h_1,h_2) = \int h_1 h_2 r dr dz\f$ для цилиндра.
 */
struct blas_dot {
    template<typename F>
    F operator()(const F* x, const F* y, int n) const {
        return blas::dot(n, x, 1, y, 1);
    }
};

/**
   \param h - этот вектор проектируем вдоль, сюда же пишем результат
   \param e - базис (ортонормированный)
   \param n - число векторов базиса
   \param m - размерность одного вектора
 */
template<typename F, typename T>
void ortoproj_along(F* h, T& e, int n, int m, F (*dot)(const F*, const F*, int n) = [](const auto* x, const auto* y, int n) {
    return blas::dot(n, x, 1, y, 1);
}) {
    // proj on:
    // h1 = sum (h,ei)/(ei,ei) ei
    // proj off:
    // h1 = h - sum (h,ei)/(ei,ei) ei

    for (int i = 0; i < n; i++) {
        auto ei_ei = dot(&e[i][0], &e[i][0], m);
        auto h_ei = dot(&h[0], &e[i][0], m);
        blas::axpy(m, -h_ei/ei_ei, &e[i][0], 1, &h[0], 1);
    }
}

/**
   Обращение матрицы общего вида методом Гаусса-Жордана с выбором главного
   элемента по столбцу. Матрица здесь маленькая -- порядка числа векторов
   подпространства, поэтому отдельного вызова lapack не заводим.
   asp::inverse_general_matrix_my умеет только double, а нам нужен и float.

   \param dest   - результат, n x n, dest[i * n + j]
   \param source - исходная матрица, n x n, не меняется
   \param n      - размерность
   \return модуль наименьшего ведущего элемента, 0 если матрица вырождена
 */
template<typename F>
F inverse_general_matrix(F* dest, const F* source, int n) {
    std::vector<F> a(source, source+n*n);
    F min_pivot = std::numeric_limits<F>::max();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dest[i*n+j] = (i == j) ? F(1) : F(0);
        }
    }

    for (int k = 0; k < n; k++) {
        int main = k;
        for (int i = k+1; i < n; i++) {
            if (std::abs(a[i*n+k]) > std::abs(a[main*n+k])) {
                main = i;
            }
        }

        F pivot = a[main*n+k];
        min_pivot = std::min(min_pivot, std::abs(pivot));
        if (pivot == F(0)) {
            return F(0);
        }

        if (main != k) {
            for (int j = 0; j < n; j++) {
                std::swap(a[k*n+j], a[main*n+j]);
                std::swap(dest[k*n+j], dest[main*n+j]);
            }
        }

        for (int j = 0; j < n; j++) {
            a[k*n+j] /= pivot;
            dest[k*n+j] /= pivot;
        }

        for (int i = 0; i < n; i++) {
            if (i == k) {
                continue;
            }
            F f = a[i*n+k];
            for (int j = 0; j < n; j++) {
                a[i*n+j] -= f*a[k*n+j];
                dest[i*n+j] -= f*dest[k*n+j];
            }
        }
    }

    return min_pivot;
}

/**
   Матрица Грама \f$(e_i, et_j)\f$

   \param g   - результат, n x n, g[i * n + j]
   \param e   - базис подпространства
   \param et  - сопряженный базис (с. в. транспонированной матрицы)
   \param n   - число векторов базиса
   \param m   - размерность одного вектора
   \param dot - скалярное произведение
 */
template<typename F, typename T, typename Dot = blas_dot>
void gramm_matrix(F* g, T& e, T& et, int n, int m, Dot dot = Dot()) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            g[i*n+j] = dot(&e[i][0], &et[j][0], m);
        }
    }
}

/**
   Матрица, обратная к матрице Грама \f$(e_i, et_j)\f$

   \param g   - результат, n x n
   \param e   - базис подпространства
   \param et  - сопряженный базис
   \param n   - число векторов базиса
   \param m   - размерность одного вектора
   \param dot - скалярное произведение
   \return модуль наименьшего ведущего элемента, 0 если базисы вырождены.
           Малое значение означает, что базисы почти ортогональны друг другу
           и проектор численно случаен -- проверять до применения.
 */
template<typename F, typename T, typename Dot = blas_dot>
F inverse_gramm_matrix(F* g, T& e, T& et, int n, int m, Dot dot = Dot()) {
    std::vector<F> gramm(n*n);
    gramm_matrix(&gramm[0], e, et, n, m, dot);
    return inverse_general_matrix(g, &gramm[0], n);
}

/**
   Коэффициенты неортогональной проекции на подпространство \f$<e_1,...,e_n>\f$.
   Вектора - собственные вектора матрицы.

   \param c   - коэффициенты, n штук
   \param h   - вектор который проецируем
   \param e   - базис подпространства на которое проецируем
   \param et  - сопряженный базис (с. в. транспонированной матрицы)
   \param ete - матрица обратная к \f$(e_i, et_j)\f$, если 0, то вычисляем
   \param n   - число векторов базиса
   \param m   - размерность одного вектора
   \param dot - скалярное произведение
 */
template<typename F, typename T, typename Dot = blas_dot>
void projection2_ext(F* c, const F* h, T& e, T& et,
                     std::type_identity_t<const F*> ete,
                     int n, int m, Dot dot = Dot())
{
    std::vector<F> rp(n);
    std::vector<F> a;
    const F* g = ete;

    if (g == 0) {
        a.resize(n*n);
        inverse_gramm_matrix(&a[0], e, et, n, m, dot);
        g = &a[0];
    }

    //rp = (h, e*_j)
    for (int i = 0; i < n; i++) {
        rp[i] = dot(h, &et[i][0], m);
    }

    //c = a * rp
    for (int i = 0; i < n; i++) {
        c[i] = 0;
        for (int j = 0; j < n; j++) {
            c[i] += rp[j]*g[j*n+i];
        }
    }
}

/**
   Проекция на подпространство \f$<e_1,...,e_n>\f$.
   Проекция неортогональная. Вектора - собственные вектора матрицы.

   Теорема: \f$A = <e_1 ... e_{nx}> <e_{nx+1} ... e_{nx+ny}>\f$,
   \f$A^t = <e^*_1 ... e^*_{nx}> <e^*_{nx+1} ... e^*_{nx+ny}>\f$
   (у транспонированной матрицы собственные значения те же, собственные
   вектора отличаются), тогда \f$<e_1 ... e_{nx}>\f$ ортогонально
   \f$<e^*_{nx+1} ... e^*_{nx+ny}>\f$, а \f$<e^*_1 ... e^*_{nx}>\f$
   ортогонально \f$<e_{nx+1} ... e_{nx+ny}>\f$.

   \param h1  - проекция
   \param h   - вектор который проецируем
   \param e   - базис подпространства на которое проецируем
   \param et  - сопряженный базис (с. в. транспонированной матрицы)
   \param ete - матрица обратная к \f$(e_i, et_j)\f$, если 0, то вычисляем
   \param n   - число векторов базиса
   \param m   - размерность одного вектора
   \param dot - скалярное произведение
 */
template<typename F, typename T, typename Dot = blas_dot>
void projection2(F* h1, const F* h, T& e, T& et,
                 std::type_identity_t<const F*> ete,
                 int n, int m, Dot dot = Dot())
{
    std::vector<F> c(n);
    projection2_ext(&c[0], h, e, et, ete, n, m, dot);

    for (int j = 0; j < m; j++) {
        h1[j] = 0;
    }
    for (int i = 0; i < n; i++) {
        blas::axpy(m, c[i], &e[i][0], 1, &h1[0], 1);
    }
}

} // namespace fdm
