#pragma once

#include "sparse.h"

#include <slu_ddefs.h>
#include <slu_sdefs.h>
#include <slu_util.h>

namespace fdm {

template<typename T>
class superlu_solver {
    SuperMatrix A, AC, L, U, B;
    csr_matrix<T> mat;
    int n;
    int nz;

    SuperLUStat_t stat;
    GlobalLU_t global;

    std::vector<int> perm_c;
    std::vector<int> perm_r;
    std::vector<int> etree;

public:
    superlu_solver() {
        StatInit(&stat);
    }
    ~superlu_solver() {
        clear();
        StatFree (&stat);
    }

    superlu_solver& operator=(csr_matrix<T>&& matrix)
    {
        clear();
        mat = std::move(matrix);
        n = static_cast<int>(mat.Ap.size()-1);
        nz = static_cast<int>(mat.Ax.size());
        init(mat);

        int panel_size;
		int relax;
        int info;

        superlu_options_t options;
		set_default_options (&options);

        perm_c.resize (n);
		perm_r.resize (n);
		etree.resize (n);

        get_perm_c (3, &A, &perm_c[0]);
		sp_preorder (&options, &A, &perm_c[0], &etree[0], &AC);

        panel_size = sp_ienv (1);
		relax      = sp_ienv (2);

        decompose(mat, &options, relax, panel_size, &info);

        verify(info == 0);
        return *this;
    }

    void solve(T* x, const T* b) {
        int info;
        memcpy (x, b, n * sizeof (T));
        init_vec(&B, x);
        do_solve(x, &info);
        verify(info == 0);
        Destroy_SuperMatrix_Store (&B);
    }

private:
    void decompose(const csr_matrix<double>&, superlu_options_t* options, int relax, int panel_size, int* info) {
		dgstrf (options, &AC, relax, panel_size, &etree[0], nullptr, 0,
                &perm_c[0], &perm_r[0], &L, &U, &global, &stat, info);
    }

    void decompose(const csr_matrix<float>&, superlu_options_t* options, int relax, int panel_size, int* info) {
		sgstrf (options, &AC, relax, panel_size, &etree[0], nullptr, 0,
                &perm_c[0], &perm_r[0], &L, &U, &global, &stat, info);
    }

    void do_solve(double* , int* info) {
        dgstrs (TRANS, &L, &U, &perm_c[0], &perm_r[0], &B, &stat, info);
    }

    void do_solve(float*, int* info) {
        sgstrs (TRANS, &L, &U, &perm_c[0], &perm_r[0], &B, &stat, info);
    }

    void init_vec(SuperMatrix* B, double* x) {
        dCreate_Dense_Matrix (B, n, 1, x, n, SLU_DN, SLU_D, SLU_GE);
    }

    void init_vec(SuperMatrix* B, float* x) {
        sCreate_Dense_Matrix (B, n, 1, x, n, SLU_DN, SLU_D, SLU_GE);
    }

    void init(csr_matrix<double>& m) {
        dCreate_CompCol_Matrix (&A,
                                n, n, nz,
                                &mat.Ax[0], &mat.Ai[0], &mat.Ap[0],
                                SLU_NC, SLU_D, SLU_GE);
    }

    void init(csr_matrix<float>& m) {
        sCreate_CompCol_Matrix (&A,
                                n, n, nz,
                                &mat.Ax[0], &mat.Ai[0], &mat.Ap[0],
                                SLU_NC, SLU_D, SLU_GE);
    }

    void clear() {

    }
};

} // namespace fdm
