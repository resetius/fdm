#include <cstring>
#include <algorithm>
#include <vector>
#include <complex>
#include <new>
#include <random>

#include "verify.h"
#include "arpack_solver.h"
#include "asp_misc.h"

namespace fdm {

using namespace asp;
using std::vector;
using std::complex;
using std::is_same;

// The bundled ARPACK is translated by f2c. Its character arguments carry
// explicit trailing lengths; ftnlen is long on the supported non-alpha ABI.
using arpack_ftnlen = long int;

// The state objects are intentionally opaque here.  Including f2c.h in a
// public project include path would make ARPACK's compatibility headers
// shadow system C headers.  Allocation and lifetime stay inside ARPACK.
extern "C" {
typedef struct arpack_snstate arpack_snstate;
typedef struct arpack_dnstate arpack_dnstate;
arpack_snstate* arpack_snstate_create(void);
void arpack_snstate_destroy(arpack_snstate*);
arpack_snstate* arpack_snstate_set(arpack_snstate*);
arpack_dnstate* arpack_dnstate_create(void);
void arpack_dnstate_destroy(arpack_dnstate*);
arpack_dnstate* arpack_dnstate_set(arpack_dnstate*);
}

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
    int* info,
    arpack_ftnlen bmat_len,
    arpack_ftnlen which_len);
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
    int* info,
    arpack_ftnlen bmat_len,
    arpack_ftnlen which_len);
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
    int* info,
    arpack_ftnlen howmany_len,
    arpack_ftnlen bmat_len,
    arpack_ftnlen which_len);
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
    int* info,
    arpack_ftnlen howmany_len,
    arpack_ftnlen bmat_len,
    arpack_ftnlen which_len);

template<typename T>
struct arpack_state_traits;

template<>
struct arpack_state_traits<float> {
    using type = arpack_snstate;

    static type* create() { return arpack_snstate_create(); }
    static void destroy(type* state) { arpack_snstate_destroy(state); }
    static type* set(type* state) { return arpack_snstate_set(state); }
};

template<>
struct arpack_state_traits<double> {
    using type = arpack_dnstate;

    static type* create() { return arpack_dnstate_create(); }
    static void destroy(type* state) { arpack_dnstate_destroy(state); }
    static type* set(type* state) { return arpack_dnstate_set(state); }
};

// Temporarily select one explicit ARPACK state.  Each reverse-communication
// call restores the caller's state before returning, so any number of solves
// can be interleaved on one thread.
template<typename T>
class arpack_state_activation {
    using traits = arpack_state_traits<T>;
    using state_type = typename traits::type;

public:
    explicit arpack_state_activation(state_type* state)
        : previous_(traits::set(state))
    { }

    ~arpack_state_activation() {
        traits::set(previous_);
    }

    arpack_state_activation(const arpack_state_activation&) = delete;
    arpack_state_activation& operator=(const arpack_state_activation&) = delete;

private:
    state_type* previous_ = nullptr;
};

// Owning scope used by the traditional blocking solve().
template<typename T>
class arpack_state_scope {
    using traits = arpack_state_traits<T>;
    using state_type = typename traits::type;

public:
    arpack_state_scope() {
        state_ = traits::create();
        if (state_ == nullptr) {
            throw std::bad_alloc();
        }
        previous_ = traits::set(state_);
    }

    ~arpack_state_scope() {
        traits::set(previous_);
        traits::destroy(state_);
    }

    arpack_state_scope(const arpack_state_scope&) = delete;
    arpack_state_scope& operator=(const arpack_state_scope&) = delete;

private:
    state_type* state_ = nullptr;
    state_type* previous_ = nullptr;
};


template<typename T>
void arpack_solver<T>::solve(
    const std::function<void(T*, const T*)>& OP,
    const std::function<void(T*, const T*)>& BX,
    std::vector<std::complex<T>>& eigenvalues,
    std::vector<std::vector<T>>& eigenvectors,
    int n_eigenvalues
    )
{
    arpack_state_scope<T> state_scope;
    last_naupd_info_ = 0;
    last_neupd_info_ = 0;
    last_nconv_ = 0;
    last_iterations_ = 0;
    eigenvalues.clear();
    eigenvectors.clear();
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
    default:
        verify(false);
        break;
    }

/*          If INFO .EQ. 0, a random initial residual vector is used. */
/*          If INFO .NE. 0, RESID contains the initial residual vector, */
/*                          possibly from a previous run. */
    int info = static_cast<int>(initial_resid_mode);

/*  NCV     Integer.  (INPUT) */
/*          Number of columns of the matrix V. NCV must satisfy the two */
/*          inequalities 2 <= NCV-NEV and NCV <= N. */
/*          This will indicate how many Arnoldi vectors are generated */
/*          at each iteration.  After the startup phase in which NEV */
/*          Arnoldi vectors are generated, the algorithm generates */
/*          approximately NCV-NEV Arnoldi vectors at each subsequent update */
/*          iteration. Most of the cost in generating each Arnoldi vector is */
/*          in the matrix-vector operation OP*x. */

    int ncv = requested_ncv > 0
        ? requested_ncv
        : std::min(2*nev+2, n);
    verify(ncv <= n);
    verify(ncv-nev >= 2);
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
                &info,
                1,
                2
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
                &info,
                1,
                2
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

    last_naupd_info_ = info;
    last_iterations_ = iparam[2];
    last_nconv_ = iparam[4];
    // ARPACK reports -8 when LAPACK cannot compute the Schur form of the
    // current Hessenberg matrix. This is a numerical failure of this
    // Arnoldi start, so callers using multiple starts can safely retry it.
    if (info == -8) {
        return;
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

    vector<T> z(n * (2*nev), 0.);

    T sigmar = 0.0; // real part of the shift
    T sigmai = 0.0; // imaginary part of the shift

    int lworkev = 3 * ncv;
    vector<T> workev(lworkev, 0.);

    vector<T> eigenvalues_real(2*nev, 0.);
    vector<T> eigenvalues_im(2*nev, 0.);

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
            &info,
            1,
            1,
            2);
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
            &info,
            1,
            1,
            2);
    }

    last_neupd_info_ = info;
    if (info == -14 && last_nconv_ == 0) {
        eigenvalues.clear();
        eigenvectors.clear();
        return;
    }
    verify(info == 0, format("*neupd: %d: ", info).c_str());
//    int nconv = std::min(iparam[4], nev);
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

template<typename T>
struct arpack_rci_session<T>::impl {
    using solver_type = arpack_solver<T>;
    using traits = arpack_state_traits<T>;
    using state_type = typename traits::type;
    using request = typename arpack_rci_session<T>::request;

    int n;
    int nev;
    int ncv;
    int ldv;
    int lworkl;
    T tol;
    int ido = 0;
    int info;
    bool finished = false;
    char bmat[2]{};
    char which[3]{};
    state_type* state = nullptr;
    vector<T> resid;
    vector<T> v;
    vector<int> iparam;
    vector<int> ipntr;
    vector<T> workd;
    vector<T> workl;
    vector<complex<T>> eigenvalues;
    vector<vector<T>> eigenvectors;
    int naupd_info = 0;
    int neupd_info = 0;
    int nconv = 0;
    int iterations = 0;

    impl(int dimension, int maximum_iterations,
         typename solver_type::Mode mode,
         typename solver_type::WhichEigenvalues which_eigenvalues,
         typename solver_type::InitialResidMode initial_resid_mode,
         T tolerance, int requested_ncv, const vector<T>& initial_resid,
         int requested_nev)
        : n(dimension)
        , nev(requested_nev)
        , ncv(requested_ncv > 0
              ? requested_ncv : std::min(2*requested_nev+2, dimension))
        , ldv(dimension)
        , lworkl(3*ncv*(ncv+6))
        , tol(tolerance)
        , info(static_cast<int>(initial_resid_mode))
        , resid(initial_resid)
        , v(static_cast<std::size_t>(ldv)*ncv, T(0))
        , iparam(11, 0)
        , ipntr(14, 0)
        , workd(3*n, T(0))
        , workl(lworkl, T(0))
    {
        verify(0 < nev && nev < n-1);
        verify(ncv <= n);
        verify(ncv-nev >= 2);

        switch (mode) {
        case solver_type::standard: strcpy(bmat, "I"); break;
        case solver_type::generalized: strcpy(bmat, "G"); break;
        default: verify(false); break;
        }
        switch (which_eigenvalues) {
        case solver_type::algebraically_largest: strcpy(which, "LA"); break;
        case solver_type::algebraically_smallest: strcpy(which, "SA"); break;
        case solver_type::largest_magnitude: strcpy(which, "LM"); break;
        case solver_type::smallest_magnitude: strcpy(which, "SM"); break;
        case solver_type::largest_real_part: strcpy(which, "LR"); break;
        case solver_type::smallest_real_part: strcpy(which, "SR"); break;
        case solver_type::largest_imaginary_part: strcpy(which, "LI"); break;
        case solver_type::smallest_imaginary_part: strcpy(which, "SI"); break;
        case solver_type::both_ends: strcpy(which, "BE"); break;
        default: verify(false); break;
        }

        iparam[0] = 1;
        iparam[2] = maximum_iterations;
        iparam[6] = static_cast<int>(mode);
        state = traits::create();
        if (state == nullptr) {
            throw std::bad_alloc();
        }
    }

    ~impl() {
        traits::destroy(state);
    }

    request advance() {
        if (finished) {
            return request::done;
        }

        {
            arpack_state_activation<T> activation(state);
            if constexpr (is_same<T,double>::value) {
                dnaupd_(
                    &ido, bmat, &n, which, &nev, &tol, resid.data(), &ncv,
                    v.data(), &ldv, iparam.data(), ipntr.data(), workd.data(),
                    workl.data(), &lworkl, &info, 1, 2);
            } else {
                snaupd_(
                    &ido, bmat, &n, which, &nev, &tol, resid.data(), &ncv,
                    v.data(), &ldv, iparam.data(), ipntr.data(), workd.data(),
                    workl.data(), &lworkl, &info, 1, 2);
            }
        }

        switch (ido) {
        case -1:
        case 1:
            return request::apply_op;
        case 2:
            return request::apply_b;
        case 99:
            finish();
            return request::done;
        case 3:
            verify(false, "ARPACK user shifts are unsupported");
            break;
        default:
            verify(false, "unknown ARPACK reverse-communication request");
            break;
        }
        return request::done;
    }

    const T* input() const {
        return &workd[ipntr[0]-1];
    }

    T* output() {
        return &workd[ipntr[1]-1];
    }

    void finish() {
        naupd_info = info;
        iterations = iparam[2];
        nconv = iparam[4];
        finished = true;
        if (info == -8) {
            return;
        }
        verify(info >= 0, format("*naupd: %d: ", info).c_str());

        int rvec = 1;
        char howmany = 'A';
        vector<int> select(ncv, 1);
        int ldz = n;
        vector<T> z(static_cast<std::size_t>(n)*(2*nev), T(0));
        T sigmar = T(0);
        T sigmai = T(0);
        vector<T> workev(3*ncv, T(0));
        vector<T> eigenvalues_real(2*nev, T(0));
        vector<T> eigenvalues_im(2*nev, T(0));

        {
            arpack_state_activation<T> activation(state);
            if constexpr (is_same<T,double>::value) {
                dneupd_(
                    &rvec, &howmany, select.data(), eigenvalues_real.data(),
                    eigenvalues_im.data(), z.data(), &ldz, &sigmar, &sigmai,
                    workev.data(), bmat, &n, which, &nev, &tol, resid.data(),
                    &ncv, v.data(), &ldv, iparam.data(), ipntr.data(),
                    workd.data(), workl.data(), &lworkl, &info, 1, 1, 2);
            } else {
                sneupd_(
                    &rvec, &howmany, select.data(), eigenvalues_real.data(),
                    eigenvalues_im.data(), z.data(), &ldz, &sigmar, &sigmai,
                    workev.data(), bmat, &n, which, &nev, &tol, resid.data(),
                    &ncv, v.data(), &ldv, iparam.data(), ipntr.data(),
                    workd.data(), workl.data(), &lworkl, &info, 1, 1, 2);
            }
        }

        neupd_info = info;
        if (info == -14 && nconv == 0) {
            return;
        }
        verify(info == 0, format("*neupd: %d: ", info).c_str());
        eigenvectors.resize(nconv);
        eigenvalues.resize(nconv);
        for (int i = 0; i < nconv; ++i) {
            eigenvectors[i].assign(z.begin()+static_cast<std::size_t>(i)*n,
                                   z.begin()+static_cast<std::size_t>(i+1)*n);
            eigenvalues[i] = complex<T>(
                eigenvalues_real[i], eigenvalues_im[i]);
        }
    }
};

template<typename T>
std::unique_ptr<arpack_rci_session<T>> arpack_solver<T>::start(
    int n_eigenvalues) const
{
    return std::unique_ptr<arpack_rci_session<T>>(
        new arpack_rci_session<T>(*this, n_eigenvalues));
}

template<typename T>
arpack_rci_session<T>::arpack_rci_session(
    const arpack_solver<T>& solver, int n_eigenvalues)
    : impl_(std::make_unique<impl>(
          solver.n, solver.maxit, solver.mode, solver.eigenvalue_of_interest,
          solver.initial_resid_mode, solver.tol, solver.requested_ncv,
          solver.resid, n_eigenvalues))
{ }

template<typename T>
arpack_rci_session<T>::~arpack_rci_session() = default;

template<typename T>
arpack_rci_session<T>::arpack_rci_session(
    arpack_rci_session&&) noexcept = default;

template<typename T>
arpack_rci_session<T>& arpack_rci_session<T>::operator=(
    arpack_rci_session&&) noexcept = default;

template<typename T>
typename arpack_rci_session<T>::request arpack_rci_session<T>::advance() {
    return impl_->advance();
}

template<typename T>
const T* arpack_rci_session<T>::input() const { return impl_->input(); }

template<typename T>
T* arpack_rci_session<T>::output() { return impl_->output(); }

template<typename T>
int arpack_rci_session<T>::size() const { return impl_->n; }

template<typename T>
int arpack_rci_session<T>::naupd_info() const { return impl_->naupd_info; }

template<typename T>
int arpack_rci_session<T>::neupd_info() const { return impl_->neupd_info; }

template<typename T>
int arpack_rci_session<T>::nconv() const { return impl_->nconv; }

template<typename T>
int arpack_rci_session<T>::iterations() const { return impl_->iterations; }

template<typename T>
const vector<complex<T>>& arpack_rci_session<T>::eigenvalues() const {
    return impl_->eigenvalues;
}

template<typename T>
const vector<vector<T>>& arpack_rci_session<T>::eigenvectors() const {
    return impl_->eigenvectors;
}

template<typename T>
void arpack_solver<T>::set_resid_random(T a, T b) {
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(a, b);
    for (int i = 0; i < n; i++) {
        resid[i] = distribution(generator);
    }
}

template class arpack_solver<double>;
template class arpack_solver<float>;
template class arpack_rci_session<double>;
template class arpack_rci_session<float>;

} // namespace fdm
