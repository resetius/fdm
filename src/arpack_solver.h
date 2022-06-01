#pragma once

#include <vector>
#include <functional>
#include <cstring>
#include <complex>

namespace fdm {

template<typename T>
class arpack_solver {
    int n;
    int maxit;

public:
/*  Mode 1:  A*x = lambda*x. */
/*           ===> OP = A  and  B = I. */

/*  Mode 2:  A*x = lambda*M*x, M symmetric positive definite */
/*           ===> OP = inv[M]*A  and  B = M. */
/*           ===> (If M can be factored see remark 3 below) */

/*  Mode 3:  A*x = lambda*M*x, M symmetric semi-definite */
/*           ===> OP = Real_Part{ inv[A - sigma*M]*M }  and  B = M. */
/*           ===> shift-and-invert mode (in real arithmetic) */
/*           If OP*x = amu*x, then */
/*           amu = 1/2 * [ 1/(lambda-sigma) + 1/(lambda-conjg(sigma)) ]. */
/*           Note: If sigma is real, i.e. imaginary part of sigma is zero; */
/*                 Real_Part{ inv[A - sigma*M]*M } == inv[A - sigma*M]*M */
/*                 amu == 1/(lambda-sigma). */

/*  Mode 4:  A*x = lambda*M*x, M symmetric semi-definite */
/*           ===> OP = Imaginary_Part{ inv[A - sigma*M]*M }  and  B = M. */
/*           ===> shift-and-invert mode (in real arithmetic) */
/*           If OP*x = amu*x, then */
/*           amu = 1/2i * [ 1/(lambda-sigma) - 1/(lambda-conjg(sigma)) ]. */

    enum Mode {
        standard = 1,
        generalized = 2
    };

    const Mode mode;

    enum WhichEigenvalues {
        algebraically_largest,
        algebraically_smallest,
        largest_magnitude,
        smallest_magnitude,
        largest_real_part,
        smallest_real_part,
        largest_imaginary_part,
        smallest_imaginary_part,
        both_ends
    };

    const WhichEigenvalues eigenvalue_of_interest;

    const T tol;

    arpack_solver(
        int n,
        int maxit,
        Mode mode,
        WhichEigenvalues eigenvalue_of_interest,
        T tol)
        : n(n)
        , maxit(maxit)
        , mode(mode)
        , eigenvalue_of_interest(eigenvalue_of_interest)
        , tol(tol)
    { }

    void solve(
        const std::function<void(T*, const T*)>& OP,
        const std::function<void(T*, const T*)>& BX,
        std::vector<std::complex<T>>& eigenvalues,
        std::vector<std::vector<T>>& eigenvectors,
        int n_eigenvalues
        );

    void solve(
        const std::function<void(T*, const T*)>& OP,
        std::vector<std::complex<T>>& eigenvalues,
        std::vector<std::vector<T>>& eigenvectors,
        int n_eigenvalues)
    {
        solve(OP, [&](T* y, const T* x) {
            memcpy(y, x, n*sizeof(T));
        }, eigenvalues, eigenvectors, n_eigenvalues);
    }
};

} // namespace fdm
