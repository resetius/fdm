#pragma once

#include <umfpack.h>
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
        for (int i = 0; i < Ap.size()-1; i++) {
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

template<typename T>
class umfpack_solver {
    double Control [UMFPACK_CONTROL];
    double Info [UMFPACK_INFO];
    void *Symbolic, *Numeric;
    csr_matrix<T> mat;
    char buf[256];

    char* format(int status) {
        snprintf(buf, sizeof(buf), "status: %d", status);
        return buf;
    }

    void prepare() {
        int status;
        if (Symbolic == nullptr)
        {
            status = umfpack_di_symbolic (mat.Ap.size()-1,
                                          mat.Ap.size()-1,
                                          &mat.Ap[0],
                                          &mat.Ai[0],
                                          &mat.Ax[0],
                                          &Symbolic, Control, Info);
            verify (status == UMFPACK_OK, format(status));
        }

        if (Numeric == nullptr)
        {
            status = umfpack_di_numeric (&mat.Ap[0],
                                         &mat.Ai[0],
                                         &mat.Ax[0],
                                         Symbolic, &Numeric, Control, Info) ;
            verify (status == UMFPACK_OK, format(status));
        }
    }

    void clear()
    {
        umfpack_di_free_symbolic (&Symbolic);
        umfpack_di_free_numeric (&Numeric);
    }

public:
    umfpack_solver()
        : Symbolic(nullptr)
        , Numeric(nullptr)
    {
        umfpack_di_defaults (Control);
    }

    umfpack_solver(csr_matrix<T>&& matrix)
        : Symbolic(nullptr)
        , Numeric(nullptr)
        , mat(std::move(matrix))
    {
        umfpack_di_defaults (Control);
        prepare();
    }

    ~umfpack_solver()
    {
        clear();
    }

    umfpack_solver& operator=(csr_matrix<T>&& matrix)
    {
        clear();
        mat = std::move(matrix);
        prepare();
        return *this;
    }

    void solve(double* x, const double* b) {
        int status;

        status = umfpack_di_solve (UMFPACK_At,
                                   &mat.Ap[0],
                                   &mat.Ai[0],
                                   &mat.Ax[0], x, b, Numeric, Control, Info);
        verify (status == UMFPACK_OK, format(status));
    }
};


} // namespace fdm
