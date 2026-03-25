#pragma once
// NSCylSycl<T> — SYCL port of NSCyl (Taylor-Couette, z-periodic).
// Fields in sycl::malloc_shared; LaplCyl3FFT2 Poisson solve on CPU.
// Coordinates: phi (azimuthal, periodic), z (axial, periodic), r (radial).

#include <sycl/sycl.hpp>
#include "lapl_cyl_sycl.h"
#include <cmath>

namespace fdm {

// ── GPU-safe 3D accessor for cylindrical fields ───────────────────────────────
// phi and z are periodic; r is bounded [r_min, r_min+r_count-1].
// Layout: ptr[ ((i%nphi+nphi)%nphi)*nz*r_count
//            + ((k%nz+nz)%nz)*r_count
//            + (j-r_min) ]
template<typename T>
struct CylAcc {
    T*  ptr;
    int nphi, nz, r_count, r_min;

    CylAcc() = default;
    CylAcc(T* p, int nphi_, int nz_, int r_min_, int r_max_)
        : ptr(p), nphi(nphi_), nz(nz_)
        , r_count(r_max_-r_min_+1), r_min(r_min_) {}

    T& operator()(int i, int k, int j) const {
        return ptr[((i%nphi+nphi)%nphi) * (nz*r_count)
                 + ((k%nz+nz)%nz)      * r_count
                 + (j - r_min)];
    }
};

// ── NSCylSycl ─────────────────────────────────────────────────────────────────
template<typename T>
class NSCylSycl {
public:
    const int   nr, nz, nphi;
    const T     r0, R, lz;
    const T     dr, dz, dphi;
    const T     dr2, dz2, dphi2;
    const T     dt, Re, U0;

private:
    sycl::queue& q;

    // ── USM field buffers ─────────────────────────────────────────────────────
    // u (radial)     : [phi=0..nphi-1][z=0..nz-1][r=-1..nr+1]
    // v (axial)      : [phi=0..nphi-1][z=0..nz-1][r=0..nr+1]
    // w (azimuthal)  : [phi=0..nphi-1][z=0..nz-1][r=0..nr+1]
    // p (pressure)   : [phi=0..nphi-1][z=0..nz-1][r=0..nr+1]
    // x, RHS, G, H   : [phi=0..nphi-1][z=0..nz-1][r=1..nr]   (interior)
    // F              : [phi=0..nphi-1][z=0..nz-1][r=0..nr]    (nr+1 radial faces)
    T *u_mem, *v_mem, *w_mem, *p_mem;
    T *x_mem, *F_mem, *G_mem, *H_mem, *RHS_mem;

    LaplCylSycl<T> lapl_solver;

    static T* shalloc(sycl::queue& q, int n) {
        return sycl::malloc_shared<T>(n, q);
    }

    // ── build CylAcc from USM block ───────────────────────────────────────────
    CylAcc<T> ua() const { return {u_mem,   nphi, nz, -1, nr+1}; }
    CylAcc<T> va() const { return {v_mem,   nphi, nz,  0, nr+1}; }
    CylAcc<T> wa() const { return {w_mem,   nphi, nz,  0, nr+1}; }
    CylAcc<T> pa() const { return {p_mem,   nphi, nz,  0, nr+1}; }
    CylAcc<T> xa() const { return {x_mem,   nphi, nz,  1, nr  }; }
    CylAcc<T> Fa() const { return {F_mem,   nphi, nz,  0, nr  }; }
    CylAcc<T> Ga() const { return {G_mem,   nphi, nz,  1, nr  }; }
    CylAcc<T> Ha() const { return {H_mem,   nphi, nz,  1, nr  }; }
    CylAcc<T> Ra() const { return {RHS_mem, nphi, nz,  1, nr  }; }

public:
    NSCylSycl(sycl::queue& q_,
              int nr_, int nz_, int nphi_,
              T r0_, T R_, T lz_,
              T U0_ = T(1), T Re_ = T(100), T dt_ = T(0.002))
        : nr(nr_), nz(nz_), nphi(nphi_)
        , r0(r0_), R(R_), lz(lz_)
        , dr((R_-r0_)/nr_), dz(lz_/nz_), dphi(T(2*M_PI)/nphi_)
        , dr2(dr*dr), dz2(dz*dz), dphi2(dphi*dphi)
        , dt(dt_), Re(Re_), U0(U0_)
        , q(q_)
        , u_mem  (shalloc(q_, nphi_*nz_*(nr_+3)))
        , v_mem  (shalloc(q_, nphi_*nz_*(nr_+2)))
        , w_mem  (shalloc(q_, nphi_*nz_*(nr_+2)))
        , p_mem  (shalloc(q_, nphi_*nz_*(nr_+2)))
        , x_mem  (shalloc(q_, nphi_*nz_*nr_))
        , F_mem  (shalloc(q_, nphi_*nz_*(nr_+1)))
        , G_mem  (shalloc(q_, nphi_*nz_*nr_))
        , H_mem  (shalloc(q_, nphi_*nz_*nr_))
        , RHS_mem(shalloc(q_, nphi_*nz_*nr_))
        , lapl_solver(q_, nr_, nz_, nphi_, r0_-dr/T(2), dr, dz, lz_)
    {
        q.memset(u_mem,   0, nphi_*nz_*(nr_+3)*sizeof(T));
        q.memset(v_mem,   0, nphi_*nz_*(nr_+2)*sizeof(T));
        q.memset(w_mem,   0, nphi_*nz_*(nr_+2)*sizeof(T));
        q.memset(p_mem,   0, nphi_*nz_*(nr_+2)*sizeof(T));
        q.memset(x_mem,   0, nphi_*nz_*nr_    *sizeof(T));
        q.memset(F_mem,   0, nphi_*nz_*(nr_+1)*sizeof(T));
        q.memset(G_mem,   0, nphi_*nz_*nr_    *sizeof(T));
        q.memset(H_mem,   0, nphi_*nz_*nr_    *sizeof(T));
        q.memset(RHS_mem, 0, nphi_*nz_*nr_    *sizeof(T));
        q.wait();
    }

    ~NSCylSycl() {
        sycl::free(u_mem,   q);  sycl::free(v_mem, q);
        sycl::free(w_mem,   q);  sycl::free(p_mem, q);
        sycl::free(x_mem,   q);  sycl::free(F_mem, q);
        sycl::free(G_mem,   q);  sycl::free(H_mem, q);
        sycl::free(RHS_mem, q);
    }

    void step() {
        kernel_init_bound();
        kernel_FGH();
        kernel_poisson_rhs();
        lapl_solver.solve(x_mem, RHS_mem);
        kernel_update_uvwp();
    }

    // Advect np particles stored as Cartesian (px,py,pz).
    // render_buf: float4[np] = {x/R, y/R, z_norm, hue} for Metal.
    void advect_particles(float* px, float* py, float* pz,
                          const float* col, float* render_buf,
                          int np, uint32_t frame)
    {
        const float fdr=(float)dr, fdz=(float)dz, fdphi=(float)dphi;
        const float fr0=(float)r0, fR=(float)R, flz=(float)lz;
        const float fdt=(float)dt;
        const int   nr_=nr, nz_=nz, nphi_=nphi;
        const CylAcc<T> ua_=ua(), va_=va(), wa_=wa();

        q.parallel_for(sycl::range<1>((size_t)np), [=](sycl::id<1> id) {
            const int ip = (int)id[0];
            float cx=px[ip], cy=py[ip], cz=pz[ip];

            // Cartesian → cylindrical
            float pr = sycl::sqrt(cx*cx + cy*cy);
            float pphi = sycl::atan2(cy, cx);   // [-π, π]
            if (pphi < 0) pphi += float(2*M_PI);
            float pz_  = cz;

            // Clamp to domain
            pr   = sycl::clamp(pr,   fr0 + fdr*0.01f, fR - fdr*0.01f);
            pz_  = sycl::fmod(pz_ + flz*4, flz);  // wrap z periodic

            // Grid indices (cell centers)
            int ir   = sycl::max(0, sycl::min(nr_-1, (int)((pr - fr0)/fdr)));
            int iz   = sycl::max(0, sycl::min(nz_-1, (int)(pz_ / fdz)));
            int iphi = sycl::max(0, sycl::min(nphi_-1, (int)(pphi / fdphi)));

            // Velocity (nearest-cell interpolation; trilinear overkill for demo)
            float ur   = (float)ua_(iphi, iz, ir);     // u at face (ir+1/2)
            float uz   = (float)va_(iphi, iz, ir+1);   // v at cell center
            float uphi = (float)wa_(iphi, iz, ir+1);   // w at cell center

            // Advance in cylindrical space
            pr   += ur    * fdt;
            pz_  += uz    * fdt;
            pphi += (pr > 1e-4f ? uphi / pr : 0.f) * fdt;

            // Wrap phi and z
            pphi = sycl::fmod(pphi + float(4*M_PI), float(2*M_PI));
            pz_  = sycl::fmod(pz_  + flz*4, flz);

            // Reseed if left radial domain
            bool out = (pr < fr0 || pr > fR);
            if (out) {
                auto hash = [](uint32_t x) -> float {
                    x = ((x>>16)^x)*0x45d9f3bu;
                    x = ((x>>16)^x)*0x45d9f3bu;
                    x ^= x>>16;
                    return (float)(x>>8) * (1.f/16777216.f);
                };
                uint32_t s = (uint32_t)ip * 2654435761u ^ frame * 1234567u;
                pr   = fr0 + (hash(s * 2246822519u) * 0.92f + 0.04f) * (fR - fr0);
                pphi = hash(s * 1234567891u) * float(2*M_PI);
                pz_  = hash(s * 3266489917u) * flz;
            }

            // Cylindrical → Cartesian
            float nx = pr * sycl::cos(pphi);
            float ny = pr * sycl::sin(pphi);
            px[ip] = nx;  py[ip] = ny;  pz[ip] = pz_;

            // Render buffer: {x/R, z_norm, y/R, hue} scaled by kZoom
            // Cartesian x/y in annular plane, z_norm vertical — for 3D rotation shader
            constexpr float kZoom = 0.8f;
            render_buf[4*ip+0] = (nx / fR) * kZoom;
            render_buf[4*ip+1] = (pz_ / flz * 2.f - 1.f) * kZoom;
            render_buf[4*ip+2] = (ny / fR) * kZoom;
            render_buf[4*ip+3] = col[ip];
        });
        q.wait();
    }

private:
    // ── Boundary conditions ───────────────────────────────────────────────────
    void kernel_init_bound() {
        auto ua_=ua(), va_=va(), wa_=wa(), pa_=pa();
        const int nr_=nr, nz_=nz, nphi_=nphi;
        const T U0_=U0, Re_=Re, dr_=dr, r0_=r0;

        // w, v at inner/outer walls; u ghost cells
        q.parallel_for(sycl::range<2>((size_t)nphi, (size_t)nz),
            [=](sycl::id<2> id) {
                int i=(int)id[0], k=(int)id[1];
                wa_(i,k,0)    = T(2)*U0_ - wa_(i,k,1);   // inner rotating
                wa_(i,k,nr_+1) = -wa_(i,k,nr_);           // outer no-slip
                va_(i,k,0)    = -va_(i,k,1);              // inner no-slip
                va_(i,k,nr_+1) = -va_(i,k,nr_);           // outer no-slip
                ua_(i,k,-1)   = ua_(i,k,1);               // ghost (div-free)
                ua_(i,k,nr_+1) = ua_(i,k,nr_-1);
            });

        // Pressure ghost cells (from radial momentum at r-walls)
        q.parallel_for(sycl::range<2>((size_t)nphi, (size_t)nz),
            [=](sycl::id<2> id) {
                int i=(int)id[0], k=(int)id[1];
                // inner wall j=0
                T r  = r0_ - dr_*T(0.5);
                T r2 = (r + T(0.5)*dr_)/r, r1 = (r - T(0.5)*dr_)/r;
                pa_(i,k,0) = pa_(i,k,1)
                    - (r2*ua_(i,k,1) - T(2)*ua_(i,k,0) + r1*ua_(i,k,-1))/Re_/dr_;
                // outer wall j=nr
                r  = r0_ + nr_*dr_ - dr_*T(0.5);
                r2 = (r + T(0.5)*dr_)/r;  r1 = (r - T(0.5)*dr_)/r;
                pa_(i,k,nr_+1) = pa_(i,k,nr_)
                    + (r2*ua_(i,k,nr_+1) - T(2)*ua_(i,k,nr_) + r1*ua_(i,k,nr_-1))/Re_/dr_;
            });
    }

    // ── FGH (momentum tendency) ───────────────────────────────────────────────
    void kernel_FGH() {
        auto ua_=ua(), va_=va(), wa_=wa();
        auto Fa_=Fa(), Ga_=Ga(), Ha_=Ha();
        const int nr_=nr, nz_=nz, nphi_=nphi;
        const T dt_=dt, Re_=Re, r0_=r0;
        const T dr_=dr, dz_=dz, dphi_=dphi;
        const T dr2_=dr2, dz2_=dz2, dphi2_=dphi2;

        // F (radial velocity tendency), staggered at r-faces j=0..nr
        q.parallel_for(sycl::range<3>((size_t)nphi, (size_t)nz, (size_t)(nr_+1)),
            [=](sycl::id<3> id) {
                int i=(int)id[0], k=(int)id[1], j=(int)id[2];
                T r  = r0_ + dr_*T(j);
                T r2 = (r + T(0.5)*dr_)/r;
                T r1 = (r - T(0.5)*dr_)/r;
                T rr = r*r;
                auto sq=[](T x){return x*x;};

                Fa_(i,k,j) = ua_(i,k,j) + dt_*(
                    (r2*ua_(i,k,j+1) - T(2)*ua_(i,k,j) + r1*ua_(i,k,j-1))/Re_/dr2_ +
                    (ua_(i,k+1,j)    - T(2)*ua_(i,k,j) + ua_(i,k-1,j)   )/Re_/dz2_ +
                    (ua_(i+1,k,j)    - T(2)*ua_(i,k,j) + ua_(i-1,k,j)   )/Re_/dphi2_/rr -
                    (r2*sq(T(0.5)*(ua_(i,k,j)+ua_(i,k,j+1))) -
                     r1*sq(T(0.5)*(ua_(i,k,j-1)+ua_(i,k,j))))/dr_ -
                    T(0.25)*((ua_(i,k,j)+ua_(i,k+1,j))*(va_(i,k,j+1)+va_(i,k,j)) -
                             (ua_(i,k-1,j)+ua_(i,k,j))*(va_(i,k-1,j+1)+va_(i,k-1,j)))/dz_ -
                    T(0.25)*((ua_(i,k,j)+ua_(i+1,k,j))*(wa_(i,k,j+1)+wa_(i,k,j)) -
                             (ua_(i-1,k,j)+ua_(i,k,j))*(wa_(i-1,k,j+1)+wa_(i-1,k,j)))/dphi_/r +
                    sq(T(0.5)*(wa_(i,k,j+1)+wa_(i,k,j)))/r - ua_(i,k,j)/rr/Re_ -
                    T(2)*(T(0.5)*(wa_(i,k,j+1)+wa_(i,k,j)) -
                          T(0.5)*(wa_(i-1,k,j+1)+wa_(i-1,k,j)))/rr/dphi_/Re_
                );
            });

        // G (axial velocity tendency), at cell centers j=1..nr
        q.parallel_for(sycl::range<3>((size_t)nphi, (size_t)nz, (size_t)nr_),
            [=](sycl::id<3> id) {
                int i=(int)id[0], k=(int)id[1], j=(int)id[2]+1;
                T r  = r0_ + dr_*T(j) - dr_*T(0.5);
                T r2 = (r + T(0.5)*dr_)/r;
                T r1 = (r - T(0.5)*dr_)/r;
                T rr = r*r;
                auto sq=[](T x){return x*x;};

                Ga_(i,k,j) = va_(i,k,j) + dt_*(
                    (r2*va_(i,k,j+1) - T(2)*va_(i,k,j) + r1*va_(i,k,j-1))/Re_/dr2_ +
                    (va_(i,k+1,j)    - T(2)*va_(i,k,j) + va_(i,k-1,j)   )/Re_/dz2_ +
                    (va_(i+1,k,j)    - T(2)*va_(i,k,j) + va_(i-1,k,j)   )/Re_/dphi2_/rr -
                    (sq(T(0.5)*(va_(i,k,j)+va_(i,k+1,j))) -
                     sq(T(0.5)*(va_(i,k-1,j)+va_(i,k,j))))/dz_ -
                    T(0.25)*(r2*(ua_(i,k,j)+ua_(i,k+1,j))*(va_(i,k,j+1)+va_(i,k,j)) -
                             r1*(ua_(i,k,j-1)+ua_(i,k+1,j-1))*(va_(i,k,j)+va_(i,k,j-1)))/dr_ -
                    T(0.25)*((wa_(i,k,j)+wa_(i,k+1,j))*(va_(i,k,j)+va_(i+1,k,j)) -
                             (wa_(i-1,k,j)+wa_(i-1,k+1,j))*(va_(i-1,k,j)+va_(i,k,j)))/dphi_/r
                );
            });

        // H (azimuthal velocity tendency), at cell centers j=1..nr
        q.parallel_for(sycl::range<3>((size_t)nphi, (size_t)nz, (size_t)nr_),
            [=](sycl::id<3> id) {
                int i=(int)id[0], k=(int)id[1], j=(int)id[2]+1;
                T r  = r0_ + dr_*T(j) - dr_*T(0.5);
                T r2 = (r + T(0.5)*dr_)/r;
                T r1 = (r - T(0.5)*dr_)/r;
                T rr = r*r;
                auto sq=[](T x){return x*x;};

                Ha_(i,k,j) = wa_(i,k,j) + dt_*(
                    (r2*wa_(i,k,j+1) - T(2)*wa_(i,k,j) + r1*wa_(i,k,j-1))/Re_/dr2_ +
                    (wa_(i,k+1,j)    - T(2)*wa_(i,k,j) + wa_(i,k-1,j)   )/Re_/dz2_ +
                    (wa_(i+1,k,j)    - T(2)*wa_(i,k,j) + wa_(i-1,k,j)   )/Re_/dphi2_/rr -
                    (sq(T(0.5)*(wa_(i+1,k,j)+wa_(i,k,j))) -
                     sq(T(0.5)*(wa_(i-1,k,j)+wa_(i,k,j))))/dphi_/r -
                    T(0.25)*(r2*(ua_(i+1,k,j)+ua_(i,k,j))*(wa_(i,k,j+1)+wa_(i,k,j)) -
                             r1*(ua_(i+1,k,j-1)+ua_(i,k,j-1))*(wa_(i,k,j)+wa_(i,k,j-1)))/dr_ -
                    T(0.25)*((wa_(i,k,j)+wa_(i,k+1,j))*(va_(i,k,j)+va_(i+1,k,j)) -
                             (wa_(i,k-1,j)+wa_(i,k,j))*(va_(i,k-1,j)+va_(i+1,k-1,j)))/dz_ -
                    wa_(i,k,j)*T(0.5)*(ua_(i+1,k,j)+ua_(i,k,j))/r - wa_(i,k,j)/rr/Re_ +
                    T(2)*(T(0.5)*(ua_(i+1,k,j)+ua_(i,k,j)) -
                          T(0.5)*(ua_(i,k,j)+ua_(i-1,k,j)))/rr/dphi_/Re_
                );
            });
    }

    // ── Poisson RHS ───────────────────────────────────────────────────────────
    void kernel_poisson_rhs() {
        auto Fa_=Fa(), Ga_=Ga(), Ha_=Ha(), Ra_=Ra(), pa_=pa();
        const int nr_=nr, nz_=nz, nphi_=nphi;
        const T dt_=dt, r0_=r0, dr_=dr, dz_=dz, dphi_=dphi;
        const T dr2_=dr2;

        q.parallel_for(sycl::range<3>((size_t)nphi, (size_t)nz, (size_t)nr_),
            [=](sycl::id<3> id) {
                int i=(int)id[0], k=(int)id[1], j=(int)id[2]+1;
                T r  = r0_ + dr_*T(j) - dr_*T(0.5);

                Ra_(i,k,j) = (((r + T(0.5)*dr_)*Fa_(i,k,j) -
                                (r - T(0.5)*dr_)*Fa_(i,k,j-1))/r/dr_ +
                               (Ga_(i,k,j) - Ga_(i,k-1,j))/dz_ +
                               (Ha_(i,k,j) - Ha_(i-1,k,j))/dphi_/r) / dt_;

                // Neumann (pressure at inner/outer walls already in p ghost cells)
                if (j <= 1)
                    Ra_(i,k,j) -= (r - dr_*T(0.5))/r * pa_(i,k,j-1)/dr2_;
                if (j >= nr_)
                    Ra_(i,k,j) -= (r + dr_*T(0.5))/r * pa_(i,k,j+1)/dr2_;
            });
    }

    // ── Update u, v, w, p ─────────────────────────────────────────────────────
    void kernel_update_uvwp() {
        auto ua_=ua(), va_=va(), wa_=wa(), pa_=pa();
        auto xa_=xa(), Fa_=Fa(), Ga_=Ga(), Ha_=Ha();
        const int nr_=nr, nz_=nz, nphi_=nphi;
        const T dt_=dt, r0_=r0, dr_=dr, dz_=dz, dphi_=dphi;

        // u: interior radial faces j=1..nr-1
        q.parallel_for(sycl::range<3>((size_t)nphi, (size_t)nz, (size_t)(nr_-1)),
            [=](sycl::id<3> id) {
                int i=(int)id[0], k=(int)id[1], j=(int)id[2]+1;
                ua_(i,k,j) = Fa_(i,k,j) - dt_/dr_*(xa_(i,k,j+1) - xa_(i,k,j));
            });

        // v: axial faces k=0..nz-1 (periodic), j=1..nr
        q.parallel_for(sycl::range<3>((size_t)nphi, (size_t)(nz_-1), (size_t)nr_),
            [=](sycl::id<3> id) {
                int i=(int)id[0], k=(int)id[1], j=(int)id[2]+1;
                va_(i,k,j) = Ga_(i,k,j) - dt_/dz_*(xa_(i,k+1,j) - xa_(i,k,j));
            });

        // w: azimuthal, j=1..nr
        q.parallel_for(sycl::range<3>((size_t)nphi, (size_t)nz, (size_t)nr_),
            [=](sycl::id<3> id) {
                int i=(int)id[0], k=(int)id[1], j=(int)id[2]+1;
                T r = r0_ + dr_*T(j) - dr_*T(0.5);
                wa_(i,k,j) = Ha_(i,k,j)
                    - dt_/dphi_/r*(xa_(i+1,k,j) - xa_(i,k,j));
            });

        // p = x (spectral pressure from Poisson solve)
        q.parallel_for(sycl::range<3>((size_t)nphi, (size_t)nz, (size_t)nr_),
            [=](sycl::id<3> id) {
                int i=(int)id[0], k=(int)id[1], j=(int)id[2]+1;
                pa_(i,k,j) = xa_(i,k,j);
            });
    }
};

} // namespace fdm
