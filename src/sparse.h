#pragma once

#include <vector>

namespace fdm {

template<typename T>
class csr_matrix {
    int prev_row = -1;

public:
    std::vector<int> Ap; // column indices of row i: Ap[i] .. Ap[i+1]
    std::vector<int> Ai; // column
    std::vector<T> Ax;

    csr_matrix() = default;
    csr_matrix(csr_matrix&& other)
        : prev_row(other.prev_row)
        , Ap(std::move(other.Ap))
        , Ai(std::move(other.Ai))
        , Ax(std::move(other.Ax))
    { }
    csr_matrix& operator=(csr_matrix&& other) {
        prev_row = other.prev_row;
        Ap.swap(other.Ap);
        Ai.swap(other.Ai);
        Ax.swap(other.Ax);
        return *this;
    }

    void add(int row, int column, T value);

    void close();

    void sort_rows();

    bool is_closed() const;

    void clear() {
        Ap.clear(); Ai.clear(); Ax.clear(); prev_row = -1;
    }

    void mul(T* r, const T* x);

    void print() const;
};

} // namespace fdm
