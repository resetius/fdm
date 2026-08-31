#pragma once
// LaplCylSycl — SYCL Poisson solver for cylindrical geometry.
// phi (periodic), z (periodic), r (Dirichlet at both walls).
// Direct O(N²) DFT in phi and z, GPU batched cyclic reduction in r.
// Compatible with LaplCyl3FFT2<T,false,tensor_flag::periodic> interface.

#include <sycl/sycl.hpp>
#include <cmath>

namespace fdm {

template<typename T>
class LaplCylSycl {
public:
    const int nr, nz, nphi;
    const int nrq;           // ceil(log2(nr+1))
    const T   r0, dr, dz, dphi;
    const T   dr2, dz2, dphi2;
    const T   lz, slz;

private:
    sycl::queue& q;

    // Eigenvalues (USM, [n])
    T* lm_phi = nullptr;  // [nphi]
    T* lm_z   = nullptr;  // [nz]

    // DFT cosine/sine tables (USM)
    // cos_phi[m * nphi + j] = cos(2π j m / nphi), m=0..nphi/2, j=0..nphi-1
    T* cos_phi = nullptr;  // [(nphi/2+1) * nphi]
    T* sin_phi = nullptr;
    T* cos_z   = nullptr;  // [(nz/2+1) * nz]
    T* sin_z   = nullptr;

    // Base tridiagonal for r (USM, [nr]): r-only, 0-based j=0..nr-1
    // L_base[j] = (r_{j+1} - dr/2) / dr² / r_{j+1},  L_base[0]=0
    // U_base[j] = (r_{j+1} + dr/2) / dr² / r_{j+1},  U_base[nr-1]=0
    // where r_{j+1} = r0 + (j+1)*dr
    T* L_base = nullptr;  // [nr]
    T* U_base = nullptr;  // [nr]

    // CR workspace (USM): indexed [mode_pair * nr + j_0based]
    // mode_pair = phi_mode * nz + z_mode,  total = nphi*nz systems
    T* D_cr = nullptr;  // [nphi*nz*nr]
    T* L_cr = nullptr;
    T* U_cr = nullptr;
    T* b_cr = nullptr;

    // Intermediate buffer for DFT pipeline
    T* tmp = nullptr;   // [nphi*nz*nr]

    // Scale factors matching LaplCyl3FFT2 / pFFT_1 / pFFT convention:
    //   forward phi: scale = dphi * sqrt(1/π)
    //   inverse phi: scale = sqrt(1/π)
    //   forward z:   scale = dz * slz  where slz = sqrt(2/lz)
    //   inverse z:   scale = slz
    // Roundtrip: scale_fwd * scale_inv * N/2 = 1 (verified for both phi and z)
    T sc_phi_f, sc_phi_i;
    T sc_z_f,   sc_z_i;

    static T* sha(sycl::queue& q, int n) {
        return sycl::malloc_shared<T>(n, q);
    }

    void init_tables() {
        const T pi = T(M_PI);

        // Eigenvalues
        for (int m = 0; m < nphi; m++)
            lm_phi[m] = T(4)/dphi2 * sq(std::sin(T(m)*pi/T(nphi)));
        for (int k = 0; k < nz; k++)
            lm_z[k]   = T(4)/dz2   * sq(std::sin(T(k)*pi/T(nz)));

        // DFT tables (cos_phi[m*nphi+j], sin_phi[m*nphi+j])
        for (int m = 0; m <= nphi/2; m++)
            for (int j = 0; j < nphi; j++) {
                T ang = T(2)*pi*T(j)*T(m)/T(nphi);
                cos_phi[m*nphi+j] = std::cos(ang);
                sin_phi[m*nphi+j] = std::sin(ang);
            }
        for (int m = 0; m <= nz/2; m++)
            for (int k = 0; k < nz; k++) {
                T ang = T(2)*pi*T(k)*T(m)/T(nz);
                cos_z[m*nz+k] = std::cos(ang);
                sin_z[m*nz+k] = std::sin(ang);
            }

        // Base tridiagonal
        for (int j = 0; j < nr; j++) {
            T r = r0 + T(j+1)*dr;
            L_base[j] = (j > 0)    ? (r - T(0.5)*dr)/dr2/r : T(0);
            U_base[j] = (j < nr-1) ? (r + T(0.5)*dr)/dr2/r : T(0);
        }
    }

    static T sq(T x) { return x*x; }

    // ── Forward DFT in phi ────────────────────────────────────────────────────
    // in [phi][z][r]  →  out [phi_mode][z][r]  packed pFFT_1 format
    void dft_phi_fwd(T* out, const T* in) {
        const int nphi_=nphi, nz_=nz, nr_=nr;
        const T* cp = cos_phi, *sp = sin_phi;
        const T  sc = sc_phi_f;
        // Each thread handles one (m, z, r); computes Re and optionally -Im
        q.parallel_for(sycl::range<3>((size_t)(nphi_/2+1), (size_t)nz_, (size_t)nr_),
            [=](sycl::id<3> id) {
                int m=(int)id[0], k=(int)id[1], j=(int)id[2];
                T re = T(0), im = T(0);
                for (int i = 0; i < nphi_; i++) {
                    T v = in[i*nz_*nr_ + k*nr_ + j];
                    re += cp[m*nphi_+i] * v;
                    im += sp[m*nphi_+i] * v;   // sum*sin = -Im(DFT[m])
                }
                out[m*nz_*nr_ + k*nr_ + j] = sc * re;
                if (m > 0 && m < nphi_/2)
                    out[(nphi_-m)*nz_*nr_ + k*nr_ + j] = sc * im;
            });
    }

    // ── Inverse DFT in phi ────────────────────────────────────────────────────
    // in [phi_mode][z][r] packed  →  out [phi][z][r]
    void idft_phi(T* out, const T* in) {
        const int nphi_=nphi, nz_=nz, nr_=nr;
        const T* cp = cos_phi, *sp = sin_phi;
        const T  sc = sc_phi_i * T(0.5);   // 0.5 from pFFT convention
        q.parallel_for(sycl::range<3>((size_t)nphi_, (size_t)nz_, (size_t)nr_),
            [=](sycl::id<3> id) {
                int i=(int)id[0], k=(int)id[1], j=(int)id[2];
                // S[0] + S[N/2]*(-1)^i + 2*sum_{m=1}^{N/2-1}(S[m]*cos + S[N-m]*sin)
                T val = in[0*nz_*nr_ + k*nr_ + j]
                      + (i%2==0 ? T(1) : T(-1)) * in[(nphi_/2)*nz_*nr_ + k*nr_ + j];
                for (int m = 1; m < nphi_/2; m++)
                    val += T(2) * (in[m*nz_*nr_+k*nr_+j]        * cp[m*nphi_+i]
                                 + in[(nphi_-m)*nz_*nr_+k*nr_+j] * sp[m*nphi_+i]);
                out[i*nz_*nr_ + k*nr_ + j] = sc * val;
            });
    }

    // ── Forward DFT in z ──────────────────────────────────────────────────────
    // in [phi_mode][z][r]  →  out [phi_mode][z_mode][r]  packed pFFT_1 format
    void dft_z_fwd(T* out, const T* in) {
        const int nphi_=nphi, nz_=nz, nr_=nr;
        const T* cz = cos_z, *sz = sin_z;
        const T  sc = sc_z_f;
        q.parallel_for(sycl::range<3>((size_t)nphi_, (size_t)(nz_/2+1), (size_t)nr_),
            [=](sycl::id<3> id) {
                int i=(int)id[0], m=(int)id[1], j=(int)id[2];
                T re = T(0), im = T(0);
                for (int k = 0; k < nz_; k++) {
                    T v = in[i*nz_*nr_ + k*nr_ + j];
                    re += cz[m*nz_+k] * v;
                    im += sz[m*nz_+k] * v;
                }
                out[i*nz_*nr_ + m*nr_ + j] = sc * re;
                if (m > 0 && m < nz_/2)
                    out[i*nz_*nr_ + (nz_-m)*nr_ + j] = sc * im;
            });
    }

    // ── Inverse DFT in z ──────────────────────────────────────────────────────
    void idft_z(T* out, const T* in) {
        const int nphi_=nphi, nz_=nz, nr_=nr;
        const T* cz = cos_z, *sz = sin_z;
        const T  sc = sc_z_i * T(0.5);
        q.parallel_for(sycl::range<3>((size_t)nphi_, (size_t)nz_, (size_t)nr_),
            [=](sycl::id<3> id) {
                int i=(int)id[0], k=(int)id[1], j=(int)id[2];
                T val = in[i*nz_*nr_ + 0*nr_ + j]
                      + (k%2==0 ? T(1) : T(-1)) * in[i*nz_*nr_ + (nz_/2)*nr_ + j];
                for (int m = 1; m < nz_/2; m++)
                    val += T(2) * (in[i*nz_*nr_+m*nr_+j]        * cz[m*nz_+k]
                                 + in[i*nz_*nr_+(nz_-m)*nr_+j]  * sz[m*nz_+k]);
                out[i*nz_*nr_ + k*nr_ + j] = sc * val;
            });
    }

    // ── Init CR workspace ─────────────────────────────────────────────────────
    // Set D_cr, L_cr, U_cr for each (phi_mode, z_mode) pair from base + eigenvalues
    void init_cr() {
        const int nphi_=nphi, nz_=nz, nr_=nr;
        const T r0_=r0, dr_=dr, dr2_=dr2;
        const T* lp=lm_phi, *lz_=lm_z;
        const T* Lb=L_base,  *Ub=U_base;
        T* Dcr=D_cr, *Lcr=L_cr, *Ucr=U_cr;
        q.parallel_for(sycl::range<3>((size_t)nphi_, (size_t)nz_, (size_t)nr_),
            [=](sycl::id<3> id) {
                int mi=(int)id[0], mk=(int)id[1], j=(int)id[2];
                int idx = mi*nz_*nr_ + mk*nr_ + j;
                T r = r0_ + T(j+1)*dr_;
                Dcr[idx] = -T(2)/dr2_ - lp[mi]/(r*r) - lz_[mk];
                Lcr[idx] = Lb[j];
                Ucr[idx] = Ub[j];
            });
    }

    // ── CR forward sweep, level l (1-indexed) ─────────────────────────────────
    void cr_fwd(int l) {
        const int nphi_=nphi, nz_=nz, nr_=nr;
        const int s = 1<<l, h = 1<<(l-1);
        // number of j positions at this level: floor((nr+1)/s)
        const int cnt = (nr+1) >> l;
        if (cnt == 0) return;
        T* Dcr=D_cr, *Lcr=L_cr, *Ucr=U_cr, *bc=b_cr;
        q.parallel_for(sycl::range<2>((size_t)(nphi_*nz_), (size_t)cnt),
            [=](sycl::id<2> id) {
                int mode = (int)id[0];
                int batch = (int)id[1];
                int j = (batch+1)*s - 1;
                if (j >= nr_) return;
                int base = mode*nr_;
                T alpha = -Lcr[base+j] / Dcr[base+j-h];
                Dcr[base+j] += alpha * Ucr[base+j-h];
                bc[base+j]  += alpha * bc[base+j-h];
                Lcr[base+j]  = alpha * Lcr[base+j-h];
                if (j+h < nr_) {
                    T gamma = -Ucr[base+j] / Dcr[base+j+h];
                    Dcr[base+j] += gamma * Lcr[base+j+h];
                    bc[base+j]  += gamma * bc[base+j+h];
                    Ucr[base+j]  = gamma * Ucr[base+j+h];
                } else {
                    Ucr[base+j] = T(0);
                }
            });
    }

    // ── CR mid step: divide apex element by its diagonal ─────────────────────
    void cr_mid() {
        const int nphi_=nphi, nz_=nz, nr_=nr, nrq_=nrq;
        T* Dcr=D_cr, *bc=b_cr;
        const int jmid = std::min((1<<(nrq_-1))-1, nr_-1);
        q.parallel_for(sycl::range<1>((size_t)(nphi_*nz_)),
            [=](sycl::id<1> id) {
                int mode = (int)id[0];
                bc[mode*nr_ + jmid] /= Dcr[mode*nr_ + jmid];
            });
    }

    // ── CR backward sweep, level l (q-1 down to 1) ───────────────────────────
    void cr_bwd(int l) {
        const int nphi_=nphi, nz_=nz, nr_=nr;
        const int s = 1<<l, h = 1<<(l-1);
        // j values: h-1, h-1+s, h-1+2s, ...
        // count: floor((nr-h)/s)+1  but guard for h > nr
        if (h > nr) return;
        const int cnt = (nr - h) / s + 1;
        T* Dcr=D_cr, *Lcr=L_cr, *Ucr=U_cr, *bc=b_cr;
        q.parallel_for(sycl::range<2>((size_t)(nphi_*nz_), (size_t)cnt),
            [=](sycl::id<2> id) {
                int mode = (int)id[0];
                int batch = (int)id[1];
                int j = (h-1) + batch*s;
                if (j >= nr_) return;
                int base = mode*nr_;
                T v = bc[base+j];
                bool has_left  = (j > 0 && j-h >= 0);
                bool has_right = (j+h < nr_);
                if (has_left)  v -= Lcr[base+j] * bc[base+j-h];
                if (has_right) v -= Ucr[base+j] * bc[base+j+h];
                bc[base+j] = v / Dcr[base+j];
            });
    }

public:
    LaplCylSycl(sycl::queue& q_,
                int nr_, int nz_, int nphi_,
                T r0_, T dr_, T dz_, T lz_)
        : nr(nr_), nz(nz_), nphi(nphi_)
        , nrq((int)std::ceil(std::log2(double(nr_+1))))
        , r0(r0_), dr(dr_), dz(dz_), dphi(T(2*M_PI)/nphi_)
        , dr2(dr_*dr_), dz2(dz_*dz_), dphi2(dphi*dphi)
        , lz(lz_), slz(std::sqrt(T(2)/lz_))
        , q(q_)
        , lm_phi (sha(q_, nphi_))
        , lm_z   (sha(q_, nz_))
        , cos_phi(sha(q_, (nphi_/2+1)*nphi_))
        , sin_phi(sha(q_, (nphi_/2+1)*nphi_))
        , cos_z  (sha(q_, (nz_/2+1)*nz_))
        , sin_z  (sha(q_, (nz_/2+1)*nz_))
        , L_base (sha(q_, nr_))
        , U_base (sha(q_, nr_))
        , D_cr   (sha(q_, nphi_*nz_*nr_))
        , L_cr   (sha(q_, nphi_*nz_*nr_))
        , U_cr   (sha(q_, nphi_*nz_*nr_))
        , b_cr   (sha(q_, nphi_*nz_*nr_))
        , tmp    (sha(q_, nphi_*nz_*nr_))
        // Scale factors:  fwd*inv*N/2 = 1
        , sc_phi_f(dphi * std::sqrt(T(1)/T(M_PI)))
        , sc_phi_i(std::sqrt(T(1)/T(M_PI)))
        , sc_z_f  (dz_  * std::sqrt(T(2)/lz_))
        , sc_z_i  (std::sqrt(T(2)/lz_))
    {
        init_tables();
    }

    ~LaplCylSycl() {
        sycl::free(lm_phi,  q); sycl::free(lm_z,   q);
        sycl::free(cos_phi, q); sycl::free(sin_phi, q);
        sycl::free(cos_z,   q); sycl::free(sin_z,   q);
        sycl::free(L_base,  q); sycl::free(U_base,  q);
        sycl::free(D_cr,    q); sycl::free(L_cr,    q);
        sycl::free(U_cr,    q); sycl::free(b_cr,    q);
        sycl::free(tmp,     q);
    }

    // solve(ans, rhs): both are T[nphi*nz*nr], layout [phi][z][r-1] (0-based r)
    void solve(T* ans, T* rhs) {
        // Copy rhs into b_cr workspace via forward FFTs
        dft_phi_fwd(tmp,   rhs);   // rhs → tmp (phi modes)
        dft_z_fwd  (b_cr,  tmp);   // tmp → b_cr (z modes)

        // Set up per-mode tridiagonal D, L, U and forward-sweep CR
        init_cr();
        for (int l = 1; l < nrq; l++) cr_fwd(l);
        cr_mid();
        for (int l = nrq-1; l >= 1; l--) cr_bwd(l);

        // Inverse FFTs: b_cr → ans
        idft_z  (tmp, b_cr);
        idft_phi(ans, tmp);
    }
};

} // namespace fdm
