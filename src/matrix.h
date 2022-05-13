#pragma once

#include <vector>
#include "verify.h"

namespace fdm {

template<typename T, bool check=true>
class matrix { };

template<typename T>
class matrix_impl {
public:
    std::vector<T> vec;
    const int rows;
    const int cols;
    const int rs;

    const int x1,x2;
    const int y1,y2;

protected:
    matrix_impl(int rows_, int cols_, int rs_ = 0)
        : rows(rows_), cols(cols_)
        , rs(rs_?rs_:cols)
        , x1(0), x2(cols-1)
        , y1(0), y2(rows-1)
    {
        vec.resize(rows*rs);
    }

    matrix_impl(int x1, int y1, int x2, int y2)
        : rows(y2-y1+1)
        , cols(x2-x1+1)
        , rs(cols)
        , x1(x1), x2(x2)
        , y1(y1), y2(y2)
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
    matrix(int x1, int y1, int x2, int y2)
        : matrix_impl<T>(x1, y1, x2, y2)
    { }

    T* operator[](int y) {
        return &this->vec[(y-this->y1)*this->rs]-this->x1;
    }
};

template<typename T>
class row {
    T* r;
    int x1, x2;

public:
    row(T* r, int x1, int x2): r(r), x1(x1), x2(x2) {}

    T& operator[](int x) {
        verify(x >= x1 && x <= x2);
        return r[x-x1];
    }
};

template<typename T>
class matrix<T,true>: public matrix_impl<T> {
public:
    matrix(int rows, int cols, int rs = 0)
        : matrix_impl<T>(rows, cols, rs)
    { }
    matrix(int x1, int y1, int x2, int y2)
        : matrix_impl<T>(x1, y1, x2, y2)
    { }

    row<T> operator[](int y) {
        verify(y >= this->y1 && y <= this->y2);
        return row<T>(&this->vec[(y-this->y1)*this->rs], this->x1, this->x2);
    }
};

} // namespace fdm
