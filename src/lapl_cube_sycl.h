#pragma once
// LaplCubeSycl — SYCL-native Poisson solver (Dirichlet BC in all 3 directions).
// Uses direct DST-I matrix evaluation (no FFT library required).
// O(N^2) per 1-D transform → suitable for N ≤ ~128.
// Layout: data[iz*ny*nx + iy*nx + ix], iz/iy/ix in 0..n-1 (interior only).

#include <sycl/sycl.hpp>
#include <cmath>

namespace fdm {

template<typename T>
class LaplCubeSycl {
public:
    const int nx, ny, nz;
    const T   dx, dy, dz;
    const T   lx, ly, lz;       // domain lengths (passed in, = n*h + h)
    const T   slx, sly, slz;    // sqrt(2/l*)

private:
    sycl::queue& q;

    // Precomputed sine tables: sin_z[k*nz + n] = sin((n+1)*(k+1)*pi/(nz+1))
    T* sin_x = nullptr;   // [nx*nx]
    T* sin_y = nullptr;   // [ny*ny]
    T* sin_z = nullptr;   // [nz*nz]

    // Eigenvalues: lm_*[i] for i = 1..n* (stored at index i, array size n+1)
    T* lm_x = nullptr;   // [nx+1]
    T* lm_y = nullptr;   // [ny+1]
    T* lm_z = nullptr;   // [nz+1]

    // Work buffer (same size as rhs/ans)
    T* work = nullptr;   // [nz*ny*nx]

    // ── helpers ──────────────────────────────────────────────────────────────
    static T* alloc_shared(sycl::queue& q, int n) {
        return sycl::malloc_shared<T>(n, q);
    }

    void init_tables() {
        // Sine tables (CPU init, then used from GPU via USM)
        for (int k = 0; k < nx; k++)
            for (int n = 0; n < nx; n++)
                sin_x[k*nx + n] = std::sin((T)(n+1)*(T)(k+1)*T(M_PI)/(T)(nx+1));
        for (int k = 0; k < ny; k++)
            for (int n = 0; n < ny; n++)
                sin_y[k*ny + n] = std::sin((T)(n+1)*(T)(k+1)*T(M_PI)/(T)(ny+1));
        for (int k = 0; k < nz; k++)
            for (int n = 0; n < nz; n++)
                sin_z[k*nz + n] = std::sin((T)(n+1)*(T)(k+1)*T(M_PI)/(T)(nz+1));

        // Eigenvalues: λ_k = 4/h² * sin²(k*π/(2*(N+1)))
        T idx2 = T(1)/(dx*dx), idy2 = T(1)/(dy*dy), idz2 = T(1)/(dz*dz);
        for (int j = 1; j <= nx; j++)
            lm_x[j] = T(4)*idx2 * std::sin((T)j*T(M_PI)*T(0.5)/(T)(nx+1))
                                 * std::sin((T)j*T(M_PI)*T(0.5)/(T)(nx+1));
        for (int k = 1; k <= ny; k++)
            lm_y[k] = T(4)*idy2 * std::sin((T)k*T(M_PI)*T(0.5)/(T)(ny+1))
                                 * std::sin((T)k*T(M_PI)*T(0.5)/(T)(ny+1));
        for (int i = 1; i <= nz; i++)
            lm_z[i] = T(4)*idz2 * std::sin((T)i*T(M_PI)*T(0.5)/(T)(nz+1))
                                 * std::sin((T)i*T(M_PI)*T(0.5)/(T)(nz+1));
    }

    // Batch DST-I along z-axis: for all (iy,ix), transform the nz-vector along z.
    // src[iz*ny*nx + iy*nx + ix], result in dst (same layout), scaled by `scale`.
    void dst_z(T* dst, const T* src, T scale) {
        const int nx_ = nx, ny_ = ny, nz_ = nz;
        const T*  sz  = sin_z;
        q.parallel_for(sycl::range<3>((size_t)nz, (size_t)ny, (size_t)nx),
            [=](sycl::id<3> id) {
                int iz = (int)id[0], iy = (int)id[1], ix = (int)id[2];
                T sum = T(0);
                const T* row = sz + iz * nz_;   // sin[iz, 0..nz-1]
                for (int n = 0; n < nz_; n++)
                    sum += row[n] * src[n * ny_ * nx_ + iy * nx_ + ix];
                dst[iz * ny_ * nx_ + iy * nx_ + ix] = sum * scale;
            });
    }

    void dst_y(T* dst, const T* src, T scale) {
        const int nx_ = nx, ny_ = ny, nz_ = nz;
        const T*  sy  = sin_y;
        q.parallel_for(sycl::range<3>((size_t)nz, (size_t)ny, (size_t)nx),
            [=](sycl::id<3> id) {
                int iz = (int)id[0], iy = (int)id[1], ix = (int)id[2];
                T sum = T(0);
                const T* row = sy + iy * ny_;
                for (int n = 0; n < ny_; n++)
                    sum += row[n] * src[iz * ny_ * nx_ + n * nx_ + ix];
                dst[iz * ny_ * nx_ + iy * nx_ + ix] = sum * scale;
            });
    }

    void dst_x(T* dst, const T* src, T scale) {
        const int nx_ = nx, ny_ = ny, nz_ = nz;
        const T*  sx  = sin_x;
        q.parallel_for(sycl::range<3>((size_t)nz, (size_t)ny, (size_t)nx),
            [=](sycl::id<3> id) {
                int iz = (int)id[0], iy = (int)id[1], ix = (int)id[2];
                T sum = T(0);
                const T* row = sx + ix * nx_;
                for (int n = 0; n < nx_; n++)
                    sum += row[n] * src[iz * ny_ * nx_ + iy * nx_ + n];
                dst[iz * ny_ * nx_ + iy * nx_ + ix] = sum * scale;
            });
    }

public:
    LaplCubeSycl(sycl::queue& q_,
                 T dx_, T dy_, T dz_,
                 T lx_, T ly_, T lz_,
                 int nx_, int ny_, int nz_)
        : nx(nx_), ny(ny_), nz(nz_)
        , dx(dx_), dy(dy_), dz(dz_)
        , lx(lx_), ly(ly_), lz(lz_)
        , slx(std::sqrt(T(2)/lx_))
        , sly(std::sqrt(T(2)/ly_))
        , slz(std::sqrt(T(2)/lz_))
        , q(q_)
        , sin_x(alloc_shared(q_, nx_*nx_))
        , sin_y(alloc_shared(q_, ny_*ny_))
        , sin_z(alloc_shared(q_, nz_*nz_))
        , lm_x(alloc_shared(q_, nx_+1))
        , lm_y(alloc_shared(q_, ny_+1))
        , lm_z(alloc_shared(q_, nz_+1))
        , work(alloc_shared(q_, nz_*ny_*nx_))
    {
        init_tables();
    }

    ~LaplCubeSycl() {
        sycl::free(sin_x, q);  sycl::free(sin_y, q);  sycl::free(sin_z, q);
        sycl::free(lm_x,  q);  sycl::free(lm_y,  q);  sycl::free(lm_z,  q);
        sycl::free(work,  q);
    }

    // ans and rhs: T[nz*ny*nx] in USM (1-based tensor layout from NSCubeSycl:
    //   element [iz=1..nz][iy=1..ny][ix=1..nx] → offset (iz-1)*ny*nx + (iy-1)*nx + (ix-1))
    void solve(T* ans, T* rhs) {
        const T fwd_z = dz * slz,  inv_z = slz;
        const T fwd_y = dy * sly,  inv_y = sly;
        const T fwd_x = dx * slx,  inv_x = slx;

        // ── Forward DST: z then y then x ─────────────────────────────────────
        dst_z(work, rhs,  fwd_z);
        dst_y(ans,  work, fwd_y);
        dst_x(work, ans,  fwd_x);

        // ── Divide by eigenvalues ─────────────────────────────────────────────
        {
            const int nx_ = nx, ny_ = ny, nz_ = nz;
            const T* lx_ = lm_x, *ly_ = lm_y, *lz_ = lm_z;
            T* work_ = work;   // local copy — avoid capturing 'this' in GPU kernel
            q.parallel_for(sycl::range<3>((size_t)nz, (size_t)ny, (size_t)nx),
                [=](sycl::id<3> id) {
                    int iz = (int)id[0], iy = (int)id[1], ix = (int)id[2];
                    T k2 = lz_[iz+1] + ly_[iy+1] + lx_[ix+1];
                    work_[iz * ny_ * nx_ + iy * nx_ + ix] /= -k2;
                });
            // fix DC mode (all-periodic would set [0][0][0]=0; Dirichlet has no DC issue)
        }

        // ── Inverse DST: x then y then z ─────────────────────────────────────
        dst_x(ans,  work, inv_x);
        dst_y(work, ans,  inv_y);
        dst_z(ans,  work, inv_z);
    }
};

} // namespace fdm
