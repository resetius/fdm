#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>

#include <type_traits>
#include <chrono>
#include <random>
#include <vector>

extern "C" {
#include <cmocka.h>
}

#include "mgsch.h"
#include "projection.h"

using namespace std;
using namespace fdm;

template<typename T>
void test_mgsch(bool euclidian) {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-10, 10);

    int n = 100;
    int m = 100;
    vector<vector<T>> vecs;
    vecs.resize(m);

    auto dot1 =  [](const T* x, const T* y, int n) {
        return blas::dot(n, x, 1, y, 1);
    };

    auto dot2 =  [](const T* x, const T* y, int n) {
        T sum = 0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                T r = 0.1 + 0.05*j;
                sum += r*x[i*10+j]*y[i*10+j];
            }
        }
        return sum;
    };

    auto dot = euclidian ? dot1 : dot2;

    for (int k = 0; k < m; k++) {
        vecs[k].resize(n);
        for (int i = 0; i < n; i++) {
            vecs[k][i] = distribution(generator);
        }
    }

    for (int k = 0; k < m; k++) {
        for (int j = 0; j < m; j++) {
            T res = dot(&vecs[k][0], &vecs[j][0], n);
            assert_true(std::fabs(res) > 1e-3);
        }
    }

    mgsch(vecs, m, n, dot);

    T tol = 1e-12;
    if constexpr(is_same<T,float>::value) {
        tol = 1e-3;
    }
    for (int k = 0; k < m; k++) {
        for (int j = 0; j < m; j++) {
            T res = dot(&vecs[k][0], &vecs[j][0], n);
            if (k == j) {
                assert_true(std::fabs(res) > tol);
            } else {
                assert_true(std::fabs(res) < tol);
            }
        }
    }
}

void test_mgsch_double(void**) {
    test_mgsch<double>(true);
}

void test_mgsch_float(void** ) {
    test_mgsch<float>(true);
}

void test_mgsch_cyl_double(void**) {
    test_mgsch<double>(false);
}

void test_mgsch_cyl_float(void** ) {
    test_mgsch<float>(false);
}

template<typename T>
void test_mgsch_checked() {
    vector<vector<T>> scaled = {
        {T(1e6), T(0), T(0)},
        {T(1e6), T(1e3), T(0)},
        {T(0), T(0), T(1e-4)}
    };
    const T tolerance = std::is_same_v<T, float> ? T(1e-5) : T(1e-12);
    const T min_relative_norm = mgsch_checked<T>(
        scaled, 3, 3, tolerance);
    assert_true(min_relative_norm > 0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const T value = blas::dot(
                3, scaled[i].data(), 1, scaled[j].data(), 1);
            const T expected = i == j ? T(1) : T(0);
            const T error = std::is_same_v<T, float> ? T(1e-5) : T(1e-14);
            assert_true(std::abs(value-expected) < error);
        }
    }

    vector<vector<T>> dependent = {
        {T(1), T(0), T(0)},
        {T(1), T(1e-14), T(0)}
    };
    assert_true(mgsch_checked<T>(dependent, 2, 3, tolerance) == T(0));
}

void test_mgsch_checked_double(void**) {
    test_mgsch_checked<double>();
}

void test_mgsch_checked_float(void**) {
    test_mgsch_checked<float>();
}

template<typename T>
void test_ortoproj_simple_along(void**) {
    vector<vector<T>> basis = {
        {1,0,0,0},
        {0,1,0,0}
    };

    vector<T> vec = {1,2,3,4};
    ortoproj_along(&vec[0], basis, 2, 4);
    assert_float_equal(vec[2], 3, 1e-15);
    assert_float_equal(vec[3], 4, 1e-15);
}

void test_ortoproj_simple_along_float(void** data) {
    test_ortoproj_simple_along<float>(data);
}

void test_ortoproj_simple_along_double(void** data) {
    test_ortoproj_simple_along<double>(data);
}

template<typename T>
void test_ortoproj_along(void**) {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-10, 10);

    int n = 20;
    int m = 100;
    vector<vector<T>> vecs;
    vecs.resize(n);

    for (int k = 0; k < n; k++) {
        vecs[k].resize(m);
        for (int i = 0; i < m; i++) {
            vecs[k][i] = distribution(generator);
        }
    }

    mgsch<T>(vecs, n, m);

    vector<T> vec(m);
    for (int i = 0; i < m; i++) {
        vec[i] = distribution(generator);
    }
    ortoproj_along(&vec[0], vecs, n, m);
    vector<T> t = vec;
    ortoproj_along(&vec[0], vecs, n, m);
    T tol = 1e-15;
    if constexpr(is_same<T,float>::value) {
        tol = 1e-4;
    }
    for (int i = 0; i < m; i++) {
        assert_float_equal(vec[i], t[i], tol);
    }
}

void test_ortoproj_along_float(void** data) {
    test_ortoproj_along<float>(data);
}

void test_ortoproj_along_double(void** data) {
    test_ortoproj_along<double>(data);
}


// Проектор строится на собственных векторах существенно ненормальной матрицы
// A = R diag(2,3,0.5) R^-1, R = [r1 r2 r3] по столбцам:
//   r1 = (1,0,0)   r2 = (1,1,0)   r3 = (0,1,1)
// сопряженный базис -- строки R^-1:
//   l1 = (1,-1,1)  l2 = (0,1,-1)  l3 = (0,0,1)
// Проектируем на неустойчивое <r1,r2>. Вектора взяты ненормированными, как их
// и отдает geev, поэтому матрица Грама не единичная.
template<typename T>
void test_projection2(bool euclidian) {
    vector<vector<T>> e  = {{2,0,0}, {3,3,0}};
    vector<vector<T>> et = {{5,-5,5}, {0,7,-7}};

    // (h1,h2) = sum w_k h1_k h2_k. Сопряженный базис для веса -- et_k / w_k,
    // так что матрица Грама и проектор обязаны получиться теми же самыми.
    static const T w[3] = {1, 2, 4};
    if (!euclidian) {
        for (int i = 0; i < 2; i++) {
            for (int k = 0; k < 3; k++) {
                et[i][k] /= w[k];
            }
        }
    }

    auto dot1 = [](const T* x, const T* y, int n) {
        return blas::dot(n, x, 1, y, 1);
    };

    auto dot2 = [](const T* x, const T* y, int n) {
        T sum = 0;
        for (int k = 0; k < n; k++) {
            sum += w[k]*x[k]*y[k];
        }
        return sum;
    };

    T tol = 1e-14;
    if constexpr(is_same<T,float>::value) {
        tol = 1e-6;
    }

    auto project = [&](const vector<T>& h) {
        vector<T> h1(3);
        if (euclidian) {
            projection2(&h1[0], &h[0], e, et, nullptr, 2, 3, dot1);
        } else {
            projection2(&h1[0], &h[0], e, et, nullptr, 2, 3, dot2);
        }
        return h1;
    };

    // P r_j = r_j для векторов подпространства
    vector<vector<T>> inside = {{1,0,0}, {1,1,0}};
    for (auto& r : inside) {
        auto h1 = project(r);
        for (int k = 0; k < 3; k++) {
            assert_true(std::fabs(h1[k]-(r[k])) < tol);
        }
    }

    // P r3 = 0: r3 принадлежит дополнительному инвариантному подпространству.
    // Евклидова ортопроекция дала бы здесь (0,1,0), см. ниже.
    vector<T> r3 = {0,1,1};
    auto h1 = project(r3);
    for (int k = 0; k < 3; k++) {
        assert_true(std::fabs(h1[k]-(0)) < tol);
    }

    // P*P = P и (h - P h) ортогонально сопряженному базису
    vector<T> h = {0.3, -1.7, 2.1};
    auto Ph = project(h);
    auto PPh = project(Ph);
    for (int k = 0; k < 3; k++) {
        assert_true(std::fabs(PPh[k]-(Ph[k])) < tol);
    }

    vector<T> rest(3);
    for (int k = 0; k < 3; k++) {
        rest[k] = h[k]-Ph[k];
    }
    for (int i = 0; i < 2; i++) {
        T res = euclidian ? dot1(&rest[0], &et[i][0], 3)
                          : dot2(&rest[0], &et[i][0], 3);
        assert_true(std::fabs(res-(0)) < tol);
    }

    // Дополнительный проектор Pm = I - Pp тоже идемпотентен, и Pp + Pm = I.
    // Так же проверялось в chafe2d_check_projection2 из main-2008.1.
    auto complement = [&](const vector<T>& q) {
        auto Pq = project(q);
        vector<T> res(3);
        for (int k = 0; k < 3; k++) {
            res[k] = q[k]-Pq[k];
        }
        return res;
    };

    auto Mh = complement(h);
    auto MMh = complement(Mh);
    for (int k = 0; k < 3; k++) {
        assert_true(std::fabs(MMh[k]-(Mh[k])) < tol);
        assert_true(std::fabs(Ph[k]+Mh[k]-(h[k])) < tol);
    }

    // Теорема из комментария к projection2: <r1,r2> ортогонально сопряженному
    // базису дополнительного подпространства, а r3 -- сопряженному базису
    // самого подпространства. Аналог check_subspaces_ortogonalization из
    // tests/test_projection.c.
    vector<T> l3 = {0,0,1};
    if (!euclidian) {
        for (int k = 0; k < 3; k++) {
            l3[k] /= w[k];
        }
    }
    auto dot_ = [&](const T* x, const T* y) {
        return euclidian ? dot1(x, y, 3) : dot2(x, y, 3);
    };
    for (auto& r : inside) {
        assert_true(std::fabs(dot_(&r[0], &l3[0])) < tol);
    }
    for (int i = 0; i < 2; i++) {
        assert_true(std::fabs(dot_(&r3[0], &et[i][0])) < tol);
    }

    // Проекция именно неортогональная. Евклидова ортопроекция на то же самое
    // подпространство <r1,r2> (ортогональный базис той же линейной оболочки)
    // оставила бы от r3 = (0,1,1) вектор (0,0,1), то есть удалила бы вместе с
    // неустойчивым и кусок устойчивого направления.
    vector<vector<T>> ortho = {{1,0,0}, {0,1,0}};
    vector<T> along = r3;
    ortoproj_along<T>(&along[0], ortho, 2, 3);
    assert_true(std::fabs(along[0]-(0)) < tol);
    assert_true(std::fabs(along[1]-(0)) < tol);
    assert_true(std::fabs(along[2]-(1)) < tol);
}

void test_projection2_double(void**) {
    test_projection2<double>(true);
}

void test_projection2_float(void**) {
    test_projection2<float>(true);
}

void test_projection2_cyl_double(void**) {
    test_projection2<double>(false);
}

void test_projection2_cyl_float(void**) {
    test_projection2<float>(false);
}

// Заранее посчитанная обратная матрица Грама должна давать тот же ответ,
// что и вычисленная на месте, а вырожденные базисы -- нулевой ведущий элемент.
template<typename T>
void test_inverse_gramm_matrix() {
    vector<vector<T>> e  = {{2,0,0}, {3,3,0}};
    vector<vector<T>> et = {{5,-5,5}, {0,7,-7}};

    vector<T> ete(4);
    T pivot = inverse_gramm_matrix(&ete[0], e, et, 2, 3);
    assert_true(pivot > 0);

    T tol = 1e-14;
    if constexpr(is_same<T,float>::value) {
        tol = 1e-6;
    }

    // (e_i, et_j) = diag(10, 21)
    assert_true(std::fabs(ete[0]-(1./10)) < tol);
    assert_true(std::fabs(ete[1]-(0)) < tol);
    assert_true(std::fabs(ete[2]-(0)) < tol);
    assert_true(std::fabs(ete[3]-(1./21)) < tol);

    vector<T> h = {0.3, -1.7, 2.1};
    vector<T> h1(3), h2(3);
    projection2(&h1[0], &h[0], e, et, nullptr, 2, 3);
    projection2(&h2[0], &h[0], e, et, &ete[0], 2, 3);
    for (int k = 0; k < 3; k++) {
        assert_true(std::fabs(h1[k]-(h2[k])) < tol);
    }

    // сопряженный базис ортогонален самому подпространству -- проектор
    // не определен, обращение обязано это заметить
    vector<vector<T>> bad = {{0,0,1}, {0,0,2}};
    assert_true(inverse_gramm_matrix(&ete[0], e, bad, 2, 3) == 0);
}

void test_inverse_gramm_matrix_double(void**) {
    test_inverse_gramm_matrix<double>();
}

void test_inverse_gramm_matrix_float(void**) {
    test_inverse_gramm_matrix<float>();
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_prestate(test_mgsch_double, nullptr),
        cmocka_unit_test_prestate(test_mgsch_float, nullptr),
        cmocka_unit_test_prestate(test_mgsch_cyl_double, nullptr),
        cmocka_unit_test_prestate(test_mgsch_cyl_float, nullptr),
        cmocka_unit_test_prestate(test_mgsch_checked_double, nullptr),
        cmocka_unit_test_prestate(test_mgsch_checked_float, nullptr),
        cmocka_unit_test_prestate(test_ortoproj_simple_along_float, nullptr),
        cmocka_unit_test_prestate(test_ortoproj_simple_along_double, nullptr),
        cmocka_unit_test_prestate(test_ortoproj_along_float, nullptr),
        cmocka_unit_test_prestate(test_ortoproj_along_double, nullptr),
        cmocka_unit_test_prestate(test_projection2_double, nullptr),
        cmocka_unit_test_prestate(test_projection2_float, nullptr),
        cmocka_unit_test_prestate(test_projection2_cyl_double, nullptr),
        cmocka_unit_test_prestate(test_projection2_cyl_float, nullptr),
        cmocka_unit_test_prestate(test_inverse_gramm_matrix_double, nullptr),
        cmocka_unit_test_prestate(test_inverse_gramm_matrix_float, nullptr),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
