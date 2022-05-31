#include <algorithm>
#include <numeric>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdio>

#include "sparse.h"
#include "verify.h"

namespace fdm {

template<typename T>
void csr_matrix<T>::add(int row, int column, T value) {
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

template<typename T>
void csr_matrix<T>::close() {
    Ap.push_back(Ai.size());
    prev_row = INT_MAX;
}

template<typename T>
bool csr_matrix<T>::is_closed() const {
    return prev_row == INT_MAX;
}

template<typename T>
void csr_matrix<T>::sort_rows() {
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

template<typename T>
void csr_matrix<T>::mul(T* r, const T* x) {
    int n = static_cast<int>(Ap.size())-1;
#pragma omp parallel for
    for (int j = 0; j < n; ++j)
    {
        const T *p = &Ax[Ap[j]];
        T rj = (T) 0.0;

        for (int i0 = Ap[j]; i0 < Ap[j + 1]; ++i0, ++p)
        {
            int i = Ai[i0];
            rj += *p * x[i];
        }

        r[j] = rj;
    }
}

template<typename T>
void csr_matrix<T>::print() const {
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

template class csr_matrix<double>;
template class csr_matrix<float>;

} // namespace fdm
