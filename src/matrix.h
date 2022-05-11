#pragma once

#include "verify.h"

namespace fdm {

template<typename T, bool check=true>
class matrix { };

template<typename T>
class matrix_impl {
protected:
    std::vector<T> vec;
    int rows;
    int cols;
    int rs;

    matrix_impl(int rows_, int cols_, int rs_ = 0): rows(rows_), cols(cols_), rs(rs_?rs_:cols)
    {
        vec.resize(rows*rs);
    }
};

template<typename T>
class matrix<T,false>: public matrix_impl<T> {
public:
    matrix(int rows, int cols, int rs = 0)
        : matrix_impl<T>(rows, cols, rs)
    { }

    T* operator[](int y) {
        return &this->vec[y*this->rs];
    }
};

template<typename T>
class row {
    T* r;
    int n;

public:
    row(T* r, int n): r(r), n(n) {}

    T& operator[](int x) {
        verify(x >=0 && x < n);
        return r[n];
    }
};

template<typename T>
class matrix<T,true>: public matrix_impl<T> {
public:
    matrix(int rows, int cols, int rs = 0)
        : matrix_impl<T>(rows, cols, rs)
    { }

    row<T> operator[](int y) {
        verify(y >= 0 && y < this->rows);
        return row<T>(&this->vec[y*this->rs], this->cols);
    }
};

} // namespace fdm
