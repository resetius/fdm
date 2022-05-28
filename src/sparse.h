#pragma once

#include "verify.h"

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

    void add(int row, int column, T value) {
        verify(row >= prev_row);
        if (row != prev_row) {
            int last = Ap.empty() ? -1 : Ap.back();
            while (static_cast<int>(Ap.size()) < row+1) {
                Ap.push_back(last);
            }
            Ap[row] = Ai.size();
        }
        Ai.push_back(column);
        Ax.push_back(value);

        prev_row = row;
    }

    void close() {
        Ap.push_back(Ai.size());
        prev_row = INT_MAX;
    }

    void sort_rows() {
        verify(is_closed());
        std::vector<int> indices(Ax.size());
        std::vector<double> ax(Ax.size());
        std::vector<int> ai(Ai.size());
        for (int i = 0; i < static_cast<int>(Ap.size())-1; i++) {
            int sz = Ap[i+1]-Ap[i];
            indices.resize(sz);
            ax.resize(sz);
            ai.resize(sz);

            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                return Ai[Ap[i]+a] < Ai[Ap[i]+b];
            });


            int k = 0;
            for (auto j : indices) {
                ai[k] = Ai[Ap[i] + j];
                ax[k] = Ax[Ap[i] + j];
                k++;
            }
            memcpy(&Ai[Ap[i]], &ai[0], sz*sizeof(int));
            memcpy(&Ax[Ap[i]], &ax[0], sz*sizeof(double));
        }
    }

    bool is_closed() const {
        return prev_row == INT_MAX;
    }

    void clear() {
        Ap.clear(); Ai.clear(); Ax.clear(); prev_row = -1;
    }

    void mul(T* r, const T* x) {
        int n = static_cast<int>(Ap.size())-1;
        for (int j = 0; j < n; ++j)
        {
            const T *p = &Ax[Ap[j]];
            T rj = (T) 0.0;
            int i0;

            for (i0 = Ap[j]; i0 < Ap[j + 1]; ++i0, ++p)
            {
                int i = Ai[i0];
                rj += *p * x[i];
            }

            r[j] = rj;
        }
    }

    void print() const {
        for (auto ap: Ap) {
            printf("%d ", ap);
        }
        printf("\n");
        for (auto ai: Ai) {
            printf("%d ", ai);
        }
        printf("\n");
        for (auto ax: Ax) {
            printf("%f ", ax);
        }
        printf("\n");
    }
};

} // namespace fdm
