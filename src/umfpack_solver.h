#pragma once

#include <umfpack.h>
#include "verify.h"

namespace fdm {

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
