#pragma once
// NSCubeSycl<T> — SYCL port of NSCube<T,check>.
// All field arrays in sycl::malloc_shared (USM); LaplCube runs on CPU.
// No Metal / SDL dependencies — include this in both demo and tests.

#include <sycl/sycl.hpp>
#include "tensor.h"
#include "lapl_cube.h"
#include "lapl_cube_sycl.h"

namespace fdm {

// GPU-safe 3-D accessor: all metadata (strides, offsets) stored by value so
// SYCL lambda capture-by-value works on Metal (GPU cannot dereference CPU
// heap pointers stored inside tensor_accessor).
template<typename T>
struct sycl_acc3 {
    T*  ptr;
    int s0, s1;     // strides: s0=z-stride (row of rows), s1=y-stride (row)
    int o0, o1, o2; // lower-bound offsets for z, y, x

    sycl_acc3() = default;
    explicit sycl_acc3(fdm::tensor<T,3,false>& t)
        : ptr(t.vec)
        , s0(t.sizes[0]), s1(t.sizes[1])
        , o0(t.offsets[0]), o1(t.offsets[2]), o2(t.offsets[4])
    {}

    struct Sub2 {
        T* ptr; int s1, o1, o2;
        struct Sub1 {
            T* ptr; int o2;
            T& operator[](int x) const { return ptr[x - o2]; }
        };
        Sub1 operator[](int y) const { return Sub1{ptr + (y - o1)*s1, o2}; }
    };
    Sub2 operator[](int z) const { return Sub2{ptr + (z - o0)*s0, s1, o1, o2}; }
};

template<typename T>
class NSCubeSycl {
public:
    using tensor3 = fdm::tensor<T, 3, false>;

    const int nx, ny, nz;
    const T   dx,  dy,  dz;
    const T   dx2, dy2, dz2;
    const T   lx,  ly,  lz;
    const T   dt,  Re,  U0;

private:
    sycl::queue& q;

    T *u_mem, *v_mem, *w_mem, *p_mem;
    T *x_mem, *F_mem, *G_mem, *H_mem, *RHS_mem;

public:
    tensor3 u, v, w, p, x, F, G, H, RHS;

    LaplCubeSycl<T> lapl_solver;

    NSCubeSycl(sycl::queue& q_,
               int nx_, int ny_, int nz_,
               T lx_ = T(2*M_PI), T ly_ = T(2*M_PI), T lz_ = T(2*M_PI),
               T U0_ = T(1), T Re_ = T(100), T dt_ = T(0.001))
        : nx(nx_), ny(ny_), nz(nz_)
        , dx(lx_/nx_), dy(ly_/ny_), dz(lz_/nz_)
        , dx2(dx*dx),  dy2(dy*dy),  dz2(dz*dz)
        , lx(lx_), ly(ly_), lz(lz_)
        , dt(dt_), Re(Re_), U0(U0_)
        , q(q_)
        , u_mem  (sycl::malloc_shared<T>((nz_+2)*(ny_+2)*(nx_+3), q_))
        , v_mem  (sycl::malloc_shared<T>((nz_+2)*(ny_+3)*(nx_+2), q_))
        , w_mem  (sycl::malloc_shared<T>((nz_+3)*(ny_+2)*(nx_+2), q_))
        , p_mem  (sycl::malloc_shared<T>((nz_+2)*(ny_+2)*(nx_+2), q_))
        , x_mem  (sycl::malloc_shared<T>( nz_   * ny_   * nx_,     q_))
        , F_mem  (sycl::malloc_shared<T>( nz_   * ny_   *(nx_+1),  q_))
        , G_mem  (sycl::malloc_shared<T>( nz_   *(ny_+1)* nx_,     q_))
        , H_mem  (sycl::malloc_shared<T>((nz_+1)* ny_   * nx_,     q_))
        , RHS_mem(sycl::malloc_shared<T>( nz_   * ny_   * nx_,     q_))
        , u  ({0,nz+1,  0,ny+1, -1,nx+1}, u_mem)
        , v  ({0,nz+1, -1,ny+1,  0,nx+1}, v_mem)
        , w  ({-1,nz+1, 0,ny+1,  0,nx+1}, w_mem)
        , p  ({0,nz+1,  0,ny+1,  0,nx+1}, p_mem)
        , x  ({1,nz,    1,ny,    1,nx},    x_mem)
        , F  ({1,nz,    1,ny,    0,nx},    F_mem)
        , G  ({1,nz,    0,ny,    1,nx},    G_mem)
        , H  ({0,nz,    1,ny,    1,nx},    H_mem)
        , RHS({1,nz,    1,ny,    1,nx},    RHS_mem)
        , lapl_solver(q_, dx, dy, dz, lx_+dx, ly_+dy, lz_+dz, nx_, ny_, nz_)
    {
        q.memset(u_mem,   0, u.size   * sizeof(T));
        q.memset(v_mem,   0, v.size   * sizeof(T));
        q.memset(w_mem,   0, w.size   * sizeof(T));
        q.memset(p_mem,   0, p.size   * sizeof(T));
        q.memset(x_mem,   0, x.size   * sizeof(T));
        q.memset(F_mem,   0, F.size   * sizeof(T));
        q.memset(G_mem,   0, G.size   * sizeof(T));
        q.memset(H_mem,   0, H.size   * sizeof(T));
        q.memset(RHS_mem, 0, RHS.size * sizeof(T));
        q.wait();
    }

    ~NSCubeSycl() {
        sycl::free(u_mem,   q);  sycl::free(v_mem,   q);
        sycl::free(w_mem,   q);  sycl::free(p_mem,   q);
        sycl::free(x_mem,   q);  sycl::free(F_mem,   q);
        sycl::free(G_mem,   q);  sycl::free(H_mem,   q);
        sycl::free(RHS_mem, q);
    }

    // Advect passive particles; render_buf = float4[]{x/hlx,y/hly,z/hlz,hue} (3D positions)
    // col[i] = fixed hue in [0,1] per particle
    void advect_particles(float* px, float* py, float* pz,
                          const float* col, float* render_buf, int np, uint32_t frame)
    {
        auto ua=sycl_acc3<T>(u), va=sycl_acc3<T>(v), wa=sycl_acc3<T>(w);
        const int nx=this->nx, ny=this->ny, nz=this->nz;
        const float fdx=(float)dx, fdy=(float)dy, fdz=(float)dz, fdt=(float)dt;
        const float hlx=(float)(lx*0.5f), hly=(float)(ly*0.5f), hlz=(float)(lz*0.5f);

        q.parallel_for(sycl::range<1>((size_t)np), [=](sycl::id<1> id) {
            const int ip = (int)id[0];
            float ppx=px[ip], ppy=py[ip], ppz=pz[ip];

            int ji = sycl::max(0, sycl::min(nx-1, (int)((ppx+hlx)/fdx)));
            int ki = sycl::max(0, sycl::min(ny-1, (int)((ppy+hly)/fdy)));
            int ii = sycl::max(0, sycl::min(nz-1, (int)((ppz+hlz)/fdz)));

            float uv = (float)ua[ii+1][ki+1][ji];
            float vv = (float)va[ii+1][ki  ][ji+1];
            float wv = (float)wa[ii  ][ki+1][ji+1];

            ppx += uv * fdt;
            ppy += vv * fdt;
            ppz += wv * fdt;

            bool out = (ppx<-hlx || ppx>hlx || ppy<-hly || ppy>hly || ppz<-hlz || ppz>hlz);
            if (out) {
                auto hash = [](uint32_t x) -> float {
                    x = ((x >> 16) ^ x) * 0x45d9f3bu;
                    x = ((x >> 16) ^ x) * 0x45d9f3bu;
                    x ^= x >> 16;
                    return (float)(x >> 8) * (1.f/16777216.f);
                };
                uint32_t s = (uint32_t)ip * 2654435761u ^ frame * 1234567u;
                ppx = (hash(s * 2246822519u) * 2.f - 1.f) * hlx * 0.92f;
                ppy = (hash(s * 1234567891u) * 2.f - 1.f) * hly * 0.92f;
                ppz = (hash(s * 3266489917u) * 2.f - 1.f) * hlz * 0.92f;
            }

            px[ip]=ppx;  py[ip]=ppy;  pz[ip]=ppz;
            render_buf[4*ip+0] = ppx / hlx;   // x: along lid motion → screen horizontal
            render_buf[4*ip+1] = ppz / hlz;   // z: vertical, lid at top → screen vertical
            render_buf[4*ip+2] = ppy / hly;   // y: depth
            render_buf[4*ip+3] = col[ip];
        });
        q.wait();
    }

    void step() {
        kernel_init_bound();
        kernel_FGH();
        kernel_poisson_rhs();
        lapl_solver.solve(x.vec, RHS.vec);
        kernel_update_uvwp();
    }

private:
    void kernel_init_bound() {
        auto ua=sycl_acc3<T>(u), va=sycl_acc3<T>(v), wa=sycl_acc3<T>(w), pa=sycl_acc3<T>(p);
        const int nx=this->nx, ny=this->ny, nz=this->nz;
        const T U0=this->U0, Re=this->Re;
        const T dx=this->dx, dy=this->dy, dz=this->dz;

        q.parallel_for(sycl::range<2>((size_t)(ny+2), (size_t)(nx+3)),
                [=](sycl::id<2> id) {
            const int k=(int)id[0], j=(int)id[1]-1;
            ua[nz+1][k][j] = T(2)*U0 - ua[nz][k][j];
        });
        q.parallel_for(sycl::range<2>((size_t)(nz+2), (size_t)(ny+2)),
                [=](sycl::id<2> id) {
            const int i=(int)id[0], k=(int)id[1];
            ua[i][k][-1]   = ua[i][k][1];
            ua[i][k][nx+1] = ua[i][k][nx-1];
        });
        q.parallel_for(sycl::range<2>((size_t)(nz+2), (size_t)(nx+2)),
                [=](sycl::id<2> id) {
            const int i=(int)id[0], j=(int)id[1];
            va[i][-1][j]   = va[i][1][j];
            va[i][ny+1][j] = va[i][ny-1][j];
        });
        q.parallel_for(sycl::range<2>((size_t)(ny+2), (size_t)(nx+2)),
                [=](sycl::id<2> id) {
            const int k=(int)id[0], j=(int)id[1];
            wa[-1][k][j]   = wa[1][k][j];
            wa[nz+1][k][j] = wa[nz-1][k][j];
        });
        q.parallel_for(sycl::range<2>((size_t)nz, (size_t)ny),
                [=](sycl::id<2> id) {
            const int i=(int)id[0]+1, k=(int)id[1]+1;
            pa[i][k][0]    = pa[i][k][1]
                - (ua[i][k][1]   -T(2)*ua[i][k][0]  +ua[i][k][-1]  )/Re/dx;
            pa[i][k][nx+1] = pa[i][k][nx]
                - (ua[i][k][nx+1]-T(2)*ua[i][k][nx] +ua[i][k][nx-1])/Re/dx;
        });
        q.parallel_for(sycl::range<2>((size_t)nz, (size_t)nx),
                [=](sycl::id<2> id) {
            const int i=(int)id[0]+1, j=(int)id[1]+1;
            pa[i][0][j]    = pa[i][1][j]
                - (va[i][1][j]   -T(2)*va[i][0][j]  +va[i][-1][j]  )/Re/dy;
            pa[i][ny+1][j] = pa[i][ny][j]
                - (va[i][ny+1][j]-T(2)*va[i][ny][j] +va[i][ny-1][j])/Re/dy;
        });
        q.parallel_for(sycl::range<2>((size_t)ny, (size_t)nx),
                [=](sycl::id<2> id) {
            const int k=(int)id[0]+1, j=(int)id[1]+1;
            pa[0][k][j]    = pa[1][k][j]
                - (wa[1][k][j]   -T(2)*wa[0][k][j]  +wa[-1][k][j]  )/Re/dz;
            pa[nz+1][k][j] = pa[nz][k][j]
                - (wa[nz+1][k][j]-T(2)*wa[nz][k][j] +wa[nz-1][k][j])/Re/dz;
        });
    }

    void kernel_FGH() {
        auto ua=sycl_acc3<T>(u), va=sycl_acc3<T>(v), wa=sycl_acc3<T>(w);
        auto Fa=sycl_acc3<T>(F), Ga=sycl_acc3<T>(G), Ha=sycl_acc3<T>(H);
        const int nx=this->nx, ny=this->ny, nz=this->nz;
        const T dt=this->dt, Re=this->Re;
        const T dx=this->dx, dy=this->dy, dz=this->dz;
        const T dx2=this->dx2, dy2=this->dy2, dz2=this->dz2;

        q.parallel_for(sycl::range<3>((size_t)nz,(size_t)ny,(size_t)(nx+1)),
                [=](sycl::id<3> id) {
            const int i=(int)id[0]+1, k=(int)id[1]+1, j=(int)id[2];
            const T uij=ua[i][k][j];
            Fa[i][k][j] = uij + dt*(
                (ua[i][k][j+1]-T(2)*uij+ua[i][k][j-1])/Re/dx2 +
                (ua[i][k+1][j]-T(2)*uij+ua[i][k-1][j])/Re/dy2 +
                (ua[i+1][k][j]-T(2)*uij+ua[i-1][k][j])/Re/dz2 -
                (T(.5)*(ua[i][k][j]+ua[i][k][j+1])*T(.5)*(ua[i][k][j]+ua[i][k][j+1]) -
                 T(.5)*(ua[i][k][j-1]+ua[i][k][j])*T(.5)*(ua[i][k][j-1]+ua[i][k][j]))/dx -
                T(.25)*((ua[i][k  ][j]+ua[i][k+1][j])*(va[i][k  ][j+1]+va[i][k  ][j]) -
                        (ua[i][k-1][j]+ua[i][k  ][j])*(va[i][k-1][j+1]+va[i][k-1][j]))/dy -
                T(.25)*((ua[i  ][k][j]+ua[i+1][k][j])*(wa[i  ][k][j+1]+wa[i  ][k][j]) -
                        (ua[i-1][k][j]+ua[i  ][k][j])*(wa[i-1][k][j+1]+wa[i-1][k][j]))/dz
            );
        });
        q.parallel_for(sycl::range<3>((size_t)nz,(size_t)(ny+1),(size_t)nx),
                [=](sycl::id<3> id) {
            const int i=(int)id[0]+1, k=(int)id[1], j=(int)id[2]+1;
            const T vij=va[i][k][j];
            Ga[i][k][j] = vij + dt*(
                (va[i][k][j+1]-T(2)*vij+va[i][k][j-1])/Re/dx2 +
                (va[i][k+1][j]-T(2)*vij+va[i][k-1][j])/Re/dy2 +
                (va[i+1][k][j]-T(2)*vij+va[i-1][k][j])/Re/dz2 -
                (T(.5)*(va[i][k][j]+va[i][k+1][j])*T(.5)*(va[i][k][j]+va[i][k+1][j]) -
                 T(.5)*(va[i][k-1][j]+va[i][k][j])*T(.5)*(va[i][k-1][j]+va[i][k][j]))/dy -
                T(.25)*((ua[i][k][j  ]+ua[i][k+1][j  ])*(va[i][k][j+1]+va[i][k][j]) -
                        (ua[i][k][j-1]+ua[i][k+1][j-1])*(va[i][k][j  ]+va[i][k][j-1]))/dx -
                T(.25)*((wa[i  ][k][j]+wa[i  ][k+1][j])*(va[i  ][k][j]+va[i+1][k  ][j]) -
                        (wa[i-1][k][j]+wa[i-1][k+1][j])*(va[i-1][k][j]+va[i  ][k  ][j]))/dz
            );
        });
        q.parallel_for(sycl::range<3>((size_t)(nz+1),(size_t)ny,(size_t)nx),
                [=](sycl::id<3> id) {
            const int i=(int)id[0], k=(int)id[1]+1, j=(int)id[2]+1;
            const T wij=wa[i][k][j];
            Ha[i][k][j] = wij + dt*(
                (wa[i][k][j+1]-T(2)*wij+wa[i][k][j-1])/Re/dx2 +
                (wa[i][k+1][j]-T(2)*wij+wa[i][k-1][j])/Re/dy2 +
                (wa[i+1][k][j]-T(2)*wij+wa[i-1][k][j])/Re/dz2 -
                (T(.5)*(wa[i+1][k][j]+wa[i][k][j])*T(.5)*(wa[i+1][k][j]+wa[i][k][j]) -
                 T(.5)*(wa[i-1][k][j]+wa[i][k][j])*T(.5)*(wa[i-1][k][j]+wa[i][k][j]))/dz -
                T(.25)*((ua[i+1][k][j  ]+ua[i][k][j  ])*(wa[i][k][j+1]+wa[i][k][j]) -
                        (ua[i+1][k][j-1]+ua[i][k][j-1])*(wa[i][k][j  ]+wa[i][k][j-1]))/dx -
                T(.25)*((wa[i][k  ][j]+wa[i][k+1][j])*(va[i][k  ][j]+va[i+1][k  ][j]) -
                        (wa[i][k-1][j]+wa[i][k  ][j])*(va[i][k-1][j]+va[i+1][k-1][j]))/dy
            );
        });
    }

    void kernel_poisson_rhs() {
        auto Fa=sycl_acc3<T>(F), Ga=sycl_acc3<T>(G), Ha=sycl_acc3<T>(H), pa=sycl_acc3<T>(p), RHSa=sycl_acc3<T>(RHS);
        const int nx=this->nx, ny=this->ny, nz=this->nz;
        const T dt=this->dt, dx=this->dx, dy=this->dy, dz=this->dz;
        const T dx2=this->dx2, dy2=this->dy2, dz2=this->dz2;

        q.parallel_for(sycl::range<3>((size_t)nz,(size_t)ny,(size_t)nx),
                [=](sycl::id<3> id) {
            const int i=(int)id[0]+1, k=(int)id[1]+1, j=(int)id[2]+1;
            T rhs = ((Fa[i][k][j]-Fa[i][k][j-1])/dx +
                     (Ga[i][k][j]-Ga[i][k-1][j])/dy +
                     (Ha[i][k][j]-Ha[i-1][k][j])/dz) / dt;
            if (i==1)  rhs -= pa[i-1][k][j]/dz2;
            if (k==1)  rhs -= pa[i][k-1][j]/dy2;
            if (j==1)  rhs -= pa[i][k][j-1]/dx2;
            if (j==nx) rhs -= pa[i][k][j+1]/dx2;
            if (k==ny) rhs -= pa[i][k+1][j]/dy2;
            if (i==nz) rhs -= pa[i+1][k][j]/dz2;
            RHSa[i][k][j] = rhs;
        });
    }

    void kernel_update_uvwp() {
        auto ua=sycl_acc3<T>(u), va=sycl_acc3<T>(v), wa=sycl_acc3<T>(w), pa=sycl_acc3<T>(p), xa=sycl_acc3<T>(x);
        auto Fa=sycl_acc3<T>(F), Ga=sycl_acc3<T>(G), Ha=sycl_acc3<T>(H);
        const int nx=this->nx, ny=this->ny, nz=this->nz;
        const T dt=this->dt, dx=this->dx, dy=this->dy, dz=this->dz;

        q.parallel_for(sycl::range<3>((size_t)nz,(size_t)ny,(size_t)(nx-1)),
                [=](sycl::id<3> id) {
            const int i=(int)id[0]+1, k=(int)id[1]+1, j=(int)id[2]+1;
            ua[i][k][j] = Fa[i][k][j] - dt/dx*(xa[i][k][j+1]-xa[i][k][j]);
        });
        q.parallel_for(sycl::range<3>((size_t)nz,(size_t)(ny-1),(size_t)nx),
                [=](sycl::id<3> id) {
            const int i=(int)id[0]+1, k=(int)id[1]+1, j=(int)id[2]+1;
            va[i][k][j] = Ga[i][k][j] - dt/dy*(xa[i][k+1][j]-xa[i][k][j]);
        });
        q.parallel_for(sycl::range<3>((size_t)(nz-1),(size_t)ny,(size_t)nx),
                [=](sycl::id<3> id) {
            const int i=(int)id[0]+1, k=(int)id[1]+1, j=(int)id[2]+1;
            wa[i][k][j] = Ha[i][k][j] - dt/dz*(xa[i+1][k][j]-xa[i][k][j]);
        });
        q.parallel_for(sycl::range<3>((size_t)nz,(size_t)ny,(size_t)nx),
                [=](sycl::id<3> id) {
            const int i=(int)id[0]+1, k=(int)id[1]+1, j=(int)id[2]+1;
            pa[i][k][j] = xa[i][k][j];
        });
    }
};

} // namespace fdm
