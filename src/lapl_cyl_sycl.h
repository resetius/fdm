#pragma once
// LaplCylSycl — SYCL Poisson solver for cylindrical geometry.
// phi and z are periodic; radial Dirichlet and Neumann matrices are supported.
// Register-resident real FFT in phi and z when their lengths are powers of
// two, the direct O(N²) DFT otherwise; GPU batched cyclic reduction in r.
// Compatible with LaplCyl3FFT2<T,false,tensor_flag::periodic> interface.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <sycl/sycl.hpp>

#include "fft/sycl/rfft_registers.h"

namespace fdm {

template<typename T>
class LaplCylSycl {
public:
    struct Profile {
        double phi_forward_ms = 0;
        double z_forward_ms = 0;
        double gauge_ms = 0;
        double cr_init_ms = 0;
        double cr_forward_ms = 0;
        double cr_backward_ms = 0;
        double cr_local_ms = 0;
        double z_inverse_ms = 0;
        double phi_inverse_ms = 0;

        double total_ms() const {
            return phi_forward_ms+z_forward_ms+gauge_ms+cr_init_ms
                +cr_forward_ms+cr_backward_ms+cr_local_ms
                +z_inverse_ms+phi_inverse_ms;
        }

        Profile& operator+=(const Profile& other) {
            phi_forward_ms += other.phi_forward_ms;
            z_forward_ms += other.z_forward_ms;
            gauge_ms += other.gauge_ms;
            cr_init_ms += other.cr_init_ms;
            cr_forward_ms += other.cr_forward_ms;
            cr_backward_ms += other.cr_backward_ms;
            cr_local_ms += other.cr_local_ms;
            z_inverse_ms += other.z_inverse_ms;
            phi_inverse_ms += other.phi_inverse_ms;
            return *this;
        }
    };

    const int nr, nz, nphi;
    const int nrq;           // ceil(log2(nr+1))
    const T   r0, dr, dz, dphi;
    const T   dr2, dz2, dphi2;
    const T   lz, slz;
    const bool radial_neumann;

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

    // A power-of-two axis gets the register FFT; anything else falls back to
    // the direct transform, which has no such restriction.
    bool fft_phi = false, fft_z = false;
    int cr_local_size = 1;
    bool use_local_cr = false;
    T* tw_phi = nullptr;
    T* tw_z   = nullptr;

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

    static sycl::queue& require_in_order(sycl::queue& queue) {
        if (!queue.has_property<sycl::property::queue::in_order>()) {
            throw std::invalid_argument(
                "LaplCylSycl requires an in-order SYCL queue");
        }
        return queue;
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

        if (fft_phi) {
            const auto t = fft_sycl::make_twiddles<T>(nphi);
            std::copy(t.begin(), t.end(), tw_phi);
        }
        if (fft_z) {
            const auto t = fft_sycl::make_twiddles<T>(nz);
            std::copy(t.begin(), t.end(), tw_z);
        }

        // Base tridiagonal
        for (int j = 0; j < nr; j++) {
            T r = r0 + T(j+1)*dr;
            L_base[j] = (j > 0)    ? (r - T(0.5)*dr)/dr2/r : T(0);
            U_base[j] = (j < nr-1) ? (r + T(0.5)*dr)/dr2/r : T(0);
        }
    }

    static T sq(T x) { return x*x; }

    static int next_power_of_two(int value) {
        int result = 1;
        while (result < value) { result *= 2; }
        return result;
    }

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
        const bool radial_neumann_=radial_neumann;
        const T* lp=lm_phi, *lz_=lm_z;
        const T* Lb=L_base,  *Ub=U_base;
        T* Dcr=D_cr, *Lcr=L_cr, *Ucr=U_cr, *bc=b_cr;
        q.parallel_for(sycl::range<3>((size_t)nphi_, (size_t)nz_, (size_t)nr_),
            [=](sycl::id<3> id) {
                int mi=(int)id[0], mk=(int)id[1], j=(int)id[2];
                int idx = mi*nz_*nr_ + mk*nr_ + j;
                T r = r0_ + T(j+1)*dr_;
                Dcr[idx] = -T(2)/dr2_ - lp[mi]/(r*r) - lz_[mk];
                Lcr[idx] = Lb[j];
                Ucr[idx] = Ub[j];
                if (radial_neumann_ && j == 0) {
                    Dcr[idx] += (r-T(0.5)*dr_)/(dr2_*r);
                }
                if (radial_neumann_ && j == nr_-1) {
                    Dcr[idx] += (r+T(0.5)*dr_)/(dr2_*r);
                }
                if (radial_neumann_
                    && mi == 0 && mk == 0 && j == nr_-1) {
                    Dcr[idx] = T(1);
                    Lcr[idx] = T(0);
                    bc[idx] = T(0);
                }
            });
    }

    // ── CR forward sweep, level l (1-indexed) ─────────────────────────────────
    void cr_fwd(int l) {
        const int nphi_=nphi, nz_=nz, nr_=nr;
        const int s = 1<<l, h = 1<<(l-1);
        const bool solve_apex = l == nrq-1;
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
                if (solve_apex) {
                    bc[base+j] /= Dcr[base+j];
                }
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

    // One independent radial system per work-group.  All cyclic-reduction
    // levels stay in local memory and therefore need only work-group barriers,
    // rather than a separate globally synchronized kernel for every level.
    void solve_cr_local() {
        const int nr_=nr, nz_=nz, nrq_=nrq;
        const int local_size=cr_local_size;
        const int mode_count=nphi*nz;
        const T r0_=r0, dr_=dr, dr2_=dr2;
        const bool radial_neumann_=radial_neumann;
        const T* lambda_phi=lm_phi;
        const T* lambda_z=lm_z;
        const T* lower_base=L_base;
        const T* upper_base=U_base;
        T* coefficients=b_cr;

        q.submit([&](sycl::handler& handler) {
            sycl::local_accessor<T, 1> diagonal(
                sycl::range<1>(local_size), handler);
            sycl::local_accessor<T, 1> lower(
                sycl::range<1>(local_size), handler);
            sycl::local_accessor<T, 1> upper(
                sycl::range<1>(local_size), handler);
            sycl::local_accessor<T, 1> rhs(
                sycl::range<1>(local_size), handler);
            handler.parallel_for(
                sycl::nd_range<1>(
                    sycl::range<1>(
                        static_cast<std::size_t>(mode_count)*local_size),
                    sycl::range<1>(local_size)),
                [=](sycl::nd_item<1> item) {
                    const int mode=static_cast<int>(item.get_group(0));
                    const int j=static_cast<int>(item.get_local_id(0));
                    const int phi_mode=mode/nz_;
                    const int z_mode=mode%nz_;

                    if (j < nr_) {
                        const int index=mode*nr_+j;
                        const T radius=r0_+T(j+1)*dr_;
                        T d=-T(2)/dr2_
                            -lambda_phi[phi_mode]/(radius*radius)
                            -lambda_z[z_mode];
                        T lo=lower_base[j];
                        const T up=upper_base[j];
                        T value=coefficients[index];
                        if (radial_neumann_ && j == 0) {
                            d += (radius-T(0.5)*dr_)/(dr2_*radius);
                        }
                        if (radial_neumann_ && j == nr_-1) {
                            d += (radius+T(0.5)*dr_)/(dr2_*radius);
                        }
                        if (radial_neumann_ && mode == 0 && j == nr_-1) {
                            d=T(1);
                            lo=T(0);
                            value=T(0);
                        }
                        diagonal[j]=d;
                        lower[j]=lo;
                        upper[j]=up;
                        rhs[j]=value;
                    } else {
                        diagonal[j]=T(1);
                        lower[j]=T(0);
                        upper[j]=T(0);
                        rhs[j]=T(0);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int level=1; level < nrq_; ++level) {
                        const int stride=1<<level;
                        const int half=stride>>1;
                        const bool active=j < nr_
                            && (j+1)%stride == 0;
                        if (active) {
                            const T alpha=-lower[j]/diagonal[j-half];
                            diagonal[j] += alpha*upper[j-half];
                            rhs[j] += alpha*rhs[j-half];
                            lower[j] = alpha*lower[j-half];
                            if (j+half < nr_) {
                                const T gamma=-upper[j]/diagonal[j+half];
                                diagonal[j] += gamma*lower[j+half];
                                rhs[j] += gamma*rhs[j+half];
                                upper[j] = gamma*upper[j+half];
                            } else {
                                upper[j]=T(0);
                            }
                            if (level == nrq_-1) {
                                rhs[j] /= diagonal[j];
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (nrq_ == 1 && j == 0) {
                        rhs[j] /= diagonal[j];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int level=nrq_-1; level >= 1; --level) {
                        const int stride=1<<level;
                        const int half=stride>>1;
                        const bool active=j < nr_
                            && j%stride == half-1;
                        if (active) {
                            T value=rhs[j];
                            if (j >= half) {
                                value -= lower[j]*rhs[j-half];
                            }
                            if (j+half < nr_) {
                                value -= upper[j]*rhs[j+half];
                            }
                            rhs[j]=value/diagonal[j];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (j < nr_) {
                        coefficients[mode*nr_+j]=rhs[j];
                    }
                });
        });
    }

public:
    LaplCylSycl(sycl::queue& q_,
                int nr_, int nz_, int nphi_,
                T r0_, T dr_, T dz_, T lz_,
                bool radial_neumann_ = false)
        : nr(nr_), nz(nz_), nphi(nphi_)
        , nrq((int)std::ceil(std::log2(double(nr_+1))))
        , r0(r0_), dr(dr_), dz(dz_), dphi(T(2*M_PI)/nphi_)
        , dr2(dr_*dr_), dz2(dz_*dz_), dphi2(dphi*dphi)
        , lz(lz_), slz(std::sqrt(T(2)/lz_))
        , radial_neumann(radial_neumann_)
        , q(require_in_order(q_))
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
        , fft_phi(fft_sycl::is_power_of_two(nphi_)
                  && nphi_ >= 8 && nphi_ <= 256)
        , fft_z  (fft_sycl::is_power_of_two(nz_)
                  && nz_ >= 8 && nz_ <= 256)
        , cr_local_size(next_power_of_two(nr_))
        , use_local_cr(false)
        , tw_phi (fft_phi ? sha(q_, 2*(nphi_/2) + 2*(nphi_/2+1)) : nullptr)
        , tw_z   (fft_z   ? sha(q_, 2*(nz_/2)   + 2*(nz_/2+1))   : nullptr)
        // Scale factors:  fwd*inv*N/2 = 1
        , sc_phi_f(dphi * std::sqrt(T(1)/T(M_PI)))
        , sc_phi_i(std::sqrt(T(1)/T(M_PI)))
        , sc_z_f  (dz_  * std::sqrt(T(2)/lz_))
        , sc_z_i  (std::sqrt(T(2)/lz_))
    {
        init_tables();
        const auto device=q.get_device();
        const auto maximum_work_group_size=device.get_info<
            sycl::info::device::max_work_group_size>();
        const auto local_memory_size=device.get_info<
            sycl::info::device::local_mem_size>();
        use_local_cr = static_cast<std::size_t>(cr_local_size)
                <= maximum_work_group_size
            && static_cast<std::size_t>(4)*cr_local_size*sizeof(T)
                <= local_memory_size;
    }

    ~LaplCylSycl() {
        sycl::free(lm_phi,  q); sycl::free(lm_z,   q);
        sycl::free(cos_phi, q); sycl::free(sin_phi, q);
        sycl::free(cos_z,   q); sycl::free(sin_z,   q);
        sycl::free(L_base,  q); sycl::free(U_base,  q);
        sycl::free(D_cr,    q); sycl::free(L_cr,    q);
        sycl::free(U_cr,    q); sycl::free(b_cr,    q);
        sycl::free(tmp,     q);
        if (tw_phi) { sycl::free(tw_phi, q); }
        if (tw_z)   { sycl::free(tw_z,   q); }
    }

private:
    // Element e of line t is at
    // (t/inner)*outer + t%inner + e*stride.
    template<bool Forward>
    bool rfft(T* out, const T* in, const T* tw, T sc, int N,
              int lines, int stride, int inner, int outer) {
#define FDM_RFFT_CASE(n)                                                      \
        case n:                                                               \
            if constexpr (Forward) {                                          \
                fft_sycl::real_forward<n>(q, out, in, tw, sc,                 \
                                          lines, stride, inner, outer);       \
            } else {                                                          \
                fft_sycl::real_inverse<n>(q, out, in, tw, sc,                 \
                                          lines, stride, inner, outer);       \
            }                                                                 \
            return true;
        switch (N) {
        FDM_RFFT_CASE(8)
        FDM_RFFT_CASE(16)
        FDM_RFFT_CASE(32)
        FDM_RFFT_CASE(64)
        FDM_RFFT_CASE(128)
        FDM_RFFT_CASE(256)
        default: return false;
        }
#undef FDM_RFFT_CASE
    }

    void transform_phi_fwd(T* out, const T* in) {
        if (!fft_phi || !rfft<true>(out, in, tw_phi, sc_phi_f, nphi,
                                    nz*nr, nz*nr, nr, nr)) {
            dft_phi_fwd(out, in);
        }
    }
    void transform_phi_inv(T* out, const T* in) {
        // The direct routine folds a 0.5 of the pFFT convention into its scale;
        // the FFT wants the factor undoubled.
        if (!fft_phi || !rfft<false>(out, in, tw_phi, sc_phi_i, nphi,
                                     nz*nr, nz*nr, nr, nr)) {
            idft_phi(out, in);
        }
    }
    void transform_z_fwd(T* out, const T* in) {
        if (!fft_z || !rfft<true>(out, in, tw_z, sc_z_f, nz,
                                  nphi*nr, nr, nr, nz*nr)) {
            dft_z_fwd(out, in);
        }
    }
    void transform_z_inv(T* out, const T* in) {
        if (!fft_z || !rfft<false>(out, in, tw_z, sc_z_i, nz,
                                   nphi*nr, nr, nr, nz*nr)) {
            idft_z(out, in);
        }
    }

public:
    // solve(ans, rhs): both are T[nphi*nz*nr], layout [phi][z][r-1] (0-based r)
    void solve(T* ans, T* rhs) {
        // Copy rhs into b_cr workspace via forward FFTs
        transform_phi_fwd(tmp,  rhs);   // rhs → tmp (phi modes)
        transform_z_fwd  (b_cr, tmp);   // tmp → b_cr (z modes)

        if (use_local_cr) {
            solve_cr_local();
        } else {
            // Fallback when one radial system does not fit in a work-group or
            // in device local memory.
            init_cr();
            for (int l = 1; l < nrq; l++) cr_fwd(l);
            for (int l = nrq-1; l >= 1; l--) cr_bwd(l);
        }

        // Inverse FFTs: b_cr → ans
        transform_z_inv  (tmp, b_cr);
        transform_phi_inv(ans, tmp);
    }

    // Diagnostic path.  A queue fence after every logical stage makes the
    // numbers portable across backends that do not expose event timestamps.
    // It intentionally changes command batching and must not be used as a
    // throughput benchmark.
    Profile solve_profiled(T* ans, T* rhs) {
        using Clock = std::chrono::steady_clock;
        Profile profile;
        q.wait_and_throw();
        const auto timed = [&](auto&& operation) {
            const auto begin = Clock::now();
            operation();
            q.wait_and_throw();
            return std::chrono::duration<double, std::milli>(
                Clock::now()-begin).count();
        };

        profile.phi_forward_ms = timed([&] {
            transform_phi_fwd(tmp, rhs);
        });
        profile.z_forward_ms = timed([&] {
            transform_z_fwd(b_cr, tmp);
        });
        if (use_local_cr) {
            profile.cr_local_ms = timed([&] { solve_cr_local(); });
        } else {
            profile.cr_init_ms = timed([&] { init_cr(); });
            profile.cr_forward_ms = timed([&] {
                for (int l = 1; l < nrq; ++l) { cr_fwd(l); }
            });
            profile.cr_backward_ms = timed([&] {
                for (int l = nrq-1; l >= 1; --l) { cr_bwd(l); }
            });
        }
        profile.z_inverse_ms = timed([&] {
            transform_z_inv(tmp, b_cr);
        });
        profile.phi_inverse_ms = timed([&] {
            transform_phi_inv(ans, tmp);
        });
        return profile;
    }

    void solve_fourier_block(T* ans, const T* rhs, int m, int l) {
        if (m < 0 || m > nphi/2 || l < 0 || l > nz/2) {
            throw std::invalid_argument(
                "Fourier block index is outside the packed range");
        }

        const int nphi_=nphi, nz_=nz, nr_=nr;
        const int phi_count = (m == 0 || 2*m == nphi_) ? 1 : 2;
        const int z_count = (l == 0 || 2*l == nz_) ? 1 : 2;
        const int phase_count = phi_count*z_count;
        const T* cp=cos_phi, *sp=sin_phi;
        const T* cz=cos_z, *sz=sin_z;
        T* temporary=tmp;
        T* coefficients=b_cr;
        const T phi_forward_scale=sc_phi_f;
        const T z_forward_scale=sc_z_f;
        const bool radial_neumann_=radial_neumann;
        const bool zero_mode = radial_neumann_ && m == 0 && l == 0;

        q.parallel_for(sycl::range<3>(
            (size_t)phi_count, (size_t)nz_, (size_t)nr_),
            [=](sycl::id<3> id) {
                const int p=(int)id[0], k=(int)id[1], j=(int)id[2];
                const int pi = p == 0 ? m : nphi_-m;
                T sum = T(0);
                for (int i = 0; i < nphi_; ++i) {
                    const T basis = p == 0
                        ? cp[m*nphi_+i] : sp[m*nphi_+i];
                    sum += basis*rhs[(i*nz_+k)*nr_+j];
                }
                temporary[(pi*nz_+k)*nr_+j] =
                    phi_forward_scale*sum;
            });

        q.parallel_for(sycl::range<2>(
            (size_t)phase_count, (size_t)nr_),
            [=](sycl::id<2> id) {
                const int phase=(int)id[0], j=(int)id[1];
                const int p=phase/z_count, zphase=phase%z_count;
                const int pi=p == 0 ? m : nphi_-m;
                const int zi=zphase == 0 ? l : nz_-l;
                T sum = T(0);
                for (int k = 0; k < nz_; ++k) {
                    const T basis = zphase == 0
                        ? cz[l*nz_+k] : sz[l*nz_+k];
                    sum += basis*temporary[(pi*nz_+k)*nr_+j];
                }
                coefficients[(pi*nz_+zi)*nr_+j] =
                    z_forward_scale*sum;
            });

        const T r0_=r0, dr_=dr, dr2_=dr2;
        const T* lambda_phi=lm_phi, *lambda_z=lm_z;
        const T* lower=L_base, *upper=U_base;
        T* diagonal=D_cr;
        q.parallel_for(sycl::range<1>((size_t)phase_count),
            [=](sycl::id<1> id) {
                const int phase=(int)id[0];
                const int p=phase/z_count, zphase=phase%z_count;
                const int pi=p == 0 ? m : nphi_-m;
                const int zi=zphase == 0 ? l : nz_-l;
                const int base=(pi*nz_+zi)*nr_;

                T r=r0_+dr_;
                diagonal[base] = -T(2)/dr2_
                    -lambda_phi[pi]/(r*r)-lambda_z[zi];
                if (radial_neumann_) {
                    diagonal[base] +=
                        (r-T(0.5)*dr_)/(dr2_*r);
                }
                for (int j = 1; j < nr_; ++j) {
                    if (zero_mode && j == nr_-1) {
                        diagonal[base+j] = T(1);
                        coefficients[base+j] = T(0);
                        continue;
                    }
                    const T factor=lower[j]/diagonal[base+j-1];
                    r=r0_+T(j+1)*dr_;
                    diagonal[base+j] = -T(2)/dr2_
                        -lambda_phi[pi]/(r*r)-lambda_z[zi]
                        -factor*upper[j-1];
                    if (radial_neumann_ && j == nr_-1) {
                        diagonal[base+j] +=
                            (r+T(0.5)*dr_)/(dr2_*r);
                    }
                    coefficients[base+j] -=
                        factor*coefficients[base+j-1];
                }
                coefficients[base+nr_-1] /= diagonal[base+nr_-1];
                for (int j = nr_-2; j >= 0; --j) {
                    coefficients[base+j] =
                        (coefficients[base+j]
                         -upper[j]*coefficients[base+j+1])
                        /diagonal[base+j];
                }
            });

        const T z_inverse_scale=sc_z_i;
        q.parallel_for(sycl::range<3>(
            (size_t)phi_count, (size_t)nz_, (size_t)nr_),
            [=](sycl::id<3> id) {
                const int p=(int)id[0], k=(int)id[1], j=(int)id[2];
                const int pi=p == 0 ? m : nphi_-m;
                T value;
                if (z_count == 1) {
                    const T sign = l == 0 || k%2 == 0 ? T(1) : T(-1);
                    value = T(0.5)*sign
                        *coefficients[(pi*nz_+l)*nr_+j];
                } else {
                    value = coefficients[(pi*nz_+l)*nr_+j]
                                *cz[l*nz_+k]
                        +coefficients[(pi*nz_+(nz_-l))*nr_+j]
                                *sz[l*nz_+k];
                }
                temporary[(pi*nz_+k)*nr_+j] = z_inverse_scale*value;
            });

        const T phi_inverse_scale=sc_phi_i;
        q.parallel_for(sycl::range<3>(
            (size_t)nphi_, (size_t)nz_, (size_t)nr_),
            [=](sycl::id<3> id) {
                const int i=(int)id[0], k=(int)id[1], j=(int)id[2];
                T value;
                if (phi_count == 1) {
                    const T sign = m == 0 || i%2 == 0 ? T(1) : T(-1);
                    value = T(0.5)*sign
                        *temporary[(m*nz_+k)*nr_+j];
                } else {
                    value = temporary[(m*nz_+k)*nr_+j]
                                *cp[m*nphi_+i]
                        +temporary[((nphi_-m)*nz_+k)*nr_+j]
                                *sp[m*nphi_+i];
                }
                ans[(i*nz_+k)*nr_+j] = phi_inverse_scale*value;
            });
    }
};

} // namespace fdm
