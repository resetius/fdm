#pragma once

#include <vector>
#include "verify.h"

namespace fdm {

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
class matrix_accessor_impl {
protected:
    T* vec;
    const int rs;
    const int x1,x2;
    const int y1,y2;

    matrix_accessor_impl(T* data, int rs, int x1, int y1, int x2, int y2)
        : vec(data)
        , rs(rs)
        , x1(x1), x2(x2)
        , y1(y1), y2(y2)
    { }
};

template<typename T, bool check=true>
class matrix_accessor {};

template<typename T>
class matrix_accessor<T,true>: public matrix_accessor_impl<T>  {
public:
    matrix_accessor(T* data, int rs, int x1, int y1, int x2, int y2)
        : matrix_accessor_impl<T>(data, rs, x1, y1, x2, y2)
    { }

    row<T> operator[](int y) {
        verify(y >= this->y1 && y <= this->y2);
        return row<T>(&this->vec[(y-this->y1)*this->rs], this->x1, this->x2);
    }

    row<const T> operator[](int y) const {
        verify(y >= this->y1 && y <= this->y2);
        return row<const T>(&this->vec[(y-this->y1)*this->rs], this->x1, this->x2);
    }
};

template<typename T>
class matrix_accessor<T,false>: public matrix_accessor_impl<T>  {
public:
    matrix_accessor(T* data, int rs, int x1, int y1, int x2, int y2)
        : matrix_accessor_impl<T>(data, rs, x1, y1, x2, y2)
    { }

    T* operator[](int y) {
        return &this->vec[(y-this->y1)*this->rs]-this->x1;
    }

    const T* operator[](int y) const {
        return &this->vec[(y-this->y1)*this->rs]-this->x1;
    }
};

template<typename T, bool check=true>
class matrix {
public:
    const int rows;
    const int cols;
    const int rs;

    const int x1,x2;
    const int y1,y2;

    std::vector<T> vec;
    matrix_accessor<T,check> acc;

public:
    matrix(int rows_, int cols_, int rs_ = 0)
        : rows(rows_), cols(cols_)
        , rs(rs_?rs_:cols)
        , x1(0), x2(cols-1)
        , y1(0), y2(rows-1)
        , vec(rows*rs)
        , acc(&vec[0], rs, x1, y1, x2, y2)
    { }

    matrix(int x1, int y1, int x2, int y2)
        : rows(y2-y1+1)
        , cols(x2-x1+1)
        , rs(cols)
        , x1(x1), x2(x2)
        , y1(y1), y2(y2)
        , vec(rows*rs)
        , acc(&vec[0], rs, x1, y1, x2, y2)
    { }

    auto operator[](int y) {
        return acc[y];
    }

    int index(int y, int x) {
        return &acc[y][x]-&acc[y1][x1];
    }

    matrix<T,check>& operator=(const matrix<T,check>& other) {
        for (int y = std::max(y1, other.y1); y < std::min(y2, other.y2); y++) {
            for (int x = std::max(x1, other.x1); x < std::min(y2, other.x2); x++) {
                (*this)[y][x] = other.acc[y][x];
            }
        }
        return *this;
    }
};

} // namespace fdm
