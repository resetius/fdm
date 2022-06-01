#include <cstring>
#include <vector>
#include <complex>

#include "verify.h"
#include "arpack_solver.h"
#include "asp_misc.h"

namespace fdm {

using namespace asp;

extern "C" void dnaupd_(
    int* ido,
    char* bmat,
    const int* n,
    char* which,
    const int* nev,
    const double* tol,
    double* resid,
    int* ncv,
    double* v,
    int* ldv,
    int* iparam,
    int* ipntr,
    double* workd,
    double* workl,
    int* lworkl,
    int* info);
extern "C" void snaupd_(
    int* ido,
    char* bmat,
    const int* n,
    char* which,
    const int* nev,
    const float* tol,
    float* resid,
    int* ncv,
    float* v,
    int* ldv,
    int* iparam,
    int* ipntr,
    float* workd,
    float* workl,
    int* lworkl,
    int* info);
extern "C" void dneupd_(
    int* rvec,
    char* howmany,
    int* select,
    double* d,
    double* di,
    double* z,
    int* ldz,
    double* sigmar,
    double* sigmai,
    double* workev,
    char* bmat,
    const int* n,
    char* which,
    const int* nev,
    const double* tol,
    double* resid,
    int* ncv,
    double* v,
    int* ldv,
    int* iparam,
    int* ipntr,
    double* workd,
    double* workl,
    int* lworkl,
    int* info);
extern "C" void sneupd_(
    int* rvec,
    char* howmany,
    int* select,
    float* d,
    float* di,
    float* z,
    int* ldz,
    float* sigmar,
    float* sigmai,
    float* workev,
    char* bmat,
    const int* n,
    char* which,
    const int* nev,
    const float* tol,
    float* resid,
    int* ncv,
    float* v,
    int* ldv,
    int* iparam,
    int* ipntr,
    float* workd,
    float* workl,
    int* lworkl,
    int* info);

using namespace std;

template<typename T>
void arpack_solver<T>::solve(
    const std::function<void(T*, const T*)>& OP,
    const std::function<void(T*, const T*)>& BX,
    std::vector<std::complex<T>>& eigenvalues,
    std::vector<std::vector<T>>& eigenvectors,
    int n_eigenvalues
    )
{
    int ido = 0;
/*  NEV     Integer.  (INPUT/OUTPUT) */
/*          Number of eigenvalues of OP to be computed. 0 < NEV < N-1. */
    int nev = n_eigenvalues;
    verify(0 < nev && nev < n-1);

/*          BMAT = 'I' -> standard eigenvalue problem A*x = lambda*x */
/*          BMAT = 'G' -> generalized eigenvalue problem A*x = lambda*B*x */
    char bmat[2];
    switch (mode) {
    case standard:
        strcpy(bmat, "I");
        break;
    case generalized:
        strcpy(bmat, "G");
        break;
    default:
        verify(false);
        break;
    }

/*          'LM' -> want the NEV eigenvalues of largest magnitude. */
/*          'SM' -> want the NEV eigenvalues of smallest magnitude. */
/*          'LR' -> want the NEV eigenvalues of largest real part. */
/*          'SR' -> want the NEV eigenvalues of smallest real part. */
/*          'LI' -> want the NEV eigenvalues of largest imaginary part. */
/*          'SI' -> want the NEV eigenvalues of smallest imaginary part. */
    char which[3];
    switch (eigenvalue_of_interest) {
    case algebraically_largest:
        strcpy(which, "LA");
        break;
    case algebraically_smallest:
        strcpy(which, "SA");
        break;
    case largest_magnitude:
        strcpy(which, "LM");
        break;
    case smallest_magnitude:
        strcpy(which, "SM");
        break;
    case largest_real_part:
        strcpy(which, "LR");
        break;
    case smallest_real_part:
        strcpy(which, "SR");
        break;
    case largest_imaginary_part:
        strcpy(which, "LI");
        break;
    case smallest_imaginary_part:
        strcpy(which, "SI");
        break;
    case both_ends:
        strcpy(which, "BE");
        break;
    }

/*          If INFO .EQ. 0, a random initial residual vector is used. */
/*          If INFO .NE. 0, RESID contains the initial residual vector, */
/*                          possibly from a previous run. */
    vector<T> resid(n, (T)1);

/*  NCV     Integer.  (INPUT) */
/*          Number of columns of the matrix V. NCV must satisfy the two */
/*          inequalities 2 <= NCV-NEV and NCV <= N. */
/*          This will indicate how many Arnoldi vectors are generated */
/*          at each iteration.  After the startup phase in which NEV */
/*          Arnoldi vectors are generated, the algorithm generates */
/*          approximately NCV-NEV Arnoldi vectors at each subsequent update */
/*          iteration. Most of the cost in generating each Arnoldi vector is */
/*          in the matrix-vector operation OP*x. */

    int ncv = (2*nev+2)<n?(2*nev+2):n;
    int ldv = n;

/*  V       Double precision array N by NCV.  (OUTPUT) */
/*          Contains the final set of Arnoldi basis vectors. */
    vector<T> v (ldv*ncv, 0);
    vector<int> iparam(11, 0);

/*          ISHIFT = 0: the shifts are provided by the user via */
/*                      reverse communication.  The real and imaginary */
/*                      parts of the NCV eigenvalues of the Hessenberg */
/*                      matrix H are returned in the part of the WORKL */
/*                      array corresponding to RITZR and RITZI. See remark */
/*                      5 below. */
/*          ISHIFT = 1: exact shifts with respect to the current */
/*                      Hessenberg matrix H.  This is equivalent to */
/*                      restarting the iteration with a starting vector */
/*                      that is a linear combination of approximate Schur */
/*                      vectors associated with the "wanted" Ritz values. */
    iparam[0] = 1;
    iparam[2] = maxit;

/*          On INPUT determines what type of eigenproblem is being solved. */
/*          Must be 1,2,3,4; See under \Description of dnaupd for the */
/*          four modes available. */
    iparam[6] = static_cast<int>(mode);
    vector<int> ipntr(14, 0);

    vector<T> workd(3*n, 0);
    int lworkl = 3*ncv*(ncv+6);
    vector<T> workl(lworkl, 0);
    int info = 1;

    while (ido != 99) {
        if constexpr (is_same<T,double>::value) {
            dnaupd_(
                &ido,
                bmat,
                &n,
                which,
                &nev,
                &tol,
                &resid[0],
                &ncv,
                &v[0],
                &ldv,
                &iparam[0],
                &ipntr[0],
                &workd[0],
                &workl[0],
                &lworkl,
                &info
                );
        } else {
            snaupd_(
                &ido,
                bmat,
                &n,
                which,
                &nev,
                &tol,
                &resid[0],
                &ncv,
                &v[0],
                &ldv,
                &iparam[0],
                &ipntr[0],
                &workd[0],
                &workl[0],
                &lworkl,
                &info
                );
        }

/*          IDO =  0: first call to the reverse communication interface */
/*          IDO = -1: compute  Y = OP * X  where */
/*                    IPNTR(1) is the pointer into WORKD for X, */
/*                    IPNTR(2) is the pointer into WORKD for Y. */
/*                    This is for the initialization phase to force the */
/*                    starting vector into the range of OP. */
/*          IDO =  1: compute  Y = OP * X  where */
/*                    IPNTR(1) is the pointer into WORKD for X, */
/*                    IPNTR(2) is the pointer into WORKD for Y. */
/*                    In mode 3 and 4, the vector B * X is already */
/*                    available in WORKD(ipntr(3)).  It does not */
/*                    need to be recomputed in forming OP * X. */
/*          IDO =  2: compute  Y = B * X  where */
/*                    IPNTR(1) is the pointer into WORKD for X, */
/*                    IPNTR(2) is the pointer into WORKD for Y. */
/*          IDO =  3: compute the IPARAM(8) real and imaginary parts */
/*                    of the shifts where INPTR(14) is the pointer */
/*                    into WORKL for placing the shifts. See Remark */
/*                    5 below. */
/*          IDO = 99: done */


        switch (ido) {
        case 99:
            break;
        case -1:
        case 1:
            OP(&workd[ipntr[2-1]-1], &workd[ipntr[1-1]-1]);
            break;
        case 2:
            BX(&workd[ipntr[2-1]-1], &workd[ipntr[1-1]-1]);
            break;
        case 3:
            verify(true, "3 unsupported");
            break;
        default:
            verify(true, "unknown ido");
            break;
        }
    }

    verify(info >= 0, format("*naupd: %d: ", info).c_str());
/*             RVEC = .FALSE.     Compute Ritz values only. */

/*             RVEC = .TRUE.      Compute the Ritz vectors or Schur vectors. */
/*                                See Remarks below. */

    int rvec = 1;

/*          = 'A': Compute NEV Ritz vectors; */
/*          = 'P': Compute NEV Schur vectors; */
    char howmny = 'A';

/*  SELECT  Logical array of dimension NCV.  (INPUT) */
/*          If HOWMNY = 'S', SELECT specifies the Ritz vectors to be */
/*          computed. To select the Ritz vector corresponding to a */
/*          Ritz value (DR(j), DI(j)), SELECT(j) must be set to .TRUE.. */
/*          If HOWMNY = 'A' or 'P', SELECT is used as internal workspace. */
    vector<int> select(ncv, 1);

    int ldz = n;

    vector<T> z(n * (nev+1), 0.);

    T sigmar = 0.0; // real part of the shift
    T sigmai = 0.0; // imaginary part of the shift

    int lworkev = 3 * ncv;
    vector<T> workev(lworkev, 0.);

    vector<T> eigenvalues_real(nev+1, 0.);
    vector<T> eigenvalues_im(nev+1, 0.);

    if constexpr (is_same<T,double>::value) {
        dneupd_(
            &rvec,
            &howmny,
            &select[0],
            &eigenvalues_real[0],
            &eigenvalues_im[0],
            &z[0],
            &ldz,
            &sigmar,
            &sigmai,
            &workev[0],
            bmat,
            &n,
            which,
            &nev,
            &tol,
            &resid[0],
            &ncv,
            &v[0],
            &ldv,
            &iparam[0],
            &ipntr[0],
            &workd[0],
            &workl[0],
            &lworkl,
            &info);
    } else {
        sneupd_(
            &rvec,
            &howmny,
            &select[0],
            &eigenvalues_real[0],
            &eigenvalues_im[0],
            &z[0],
            &ldz,
            &sigmar,
            &sigmai,
            &workev[0],
            bmat,
            &n,
            which,
            &nev,
            &tol,
            &resid[0],
            &ncv,
            &v[0],
            &ldv,
            &iparam[0],
            &ipntr[0],
            &workd[0],
            &workl[0],
            &lworkl,
            &info);
    }

    verify(info == 0, format("*neupd: %d: ", info).c_str());
    int nconv = iparam[4];

    eigenvectors.resize(nconv);
    for (int i = 0; i < nconv; i++) {
        eigenvectors[i].resize(n);
        memcpy(&eigenvectors[i][0], &z[i*n], n*sizeof(T));
    }
    eigenvalues.resize(nconv);
    for (int i = 0; i < nconv; i++) {
        eigenvalues[i] = complex<T>(eigenvalues_real[i], eigenvalues_im[i]);
    }
}

template class arpack_solver<double>;
template class arpack_solver<float>;

} // namespace fdm
