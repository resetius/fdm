// Numerical inf-sup (LBB) test for the staggered NSCyl discretization on a
// cylindrical annulus.
//
// The divergence and the discrete gradient commute with shifts in phi and z,
// so the Schur complement is block diagonal in the Fourier cells (m,l) and
// splits into one small generalized eigenproblem per cell:
//
//     beta^2 = min over (m,l) of the smallest nonzero eigenvalue of
//              M_p^{1/2} D A^{-1} D^T M_p^{1/2},
//
// with A the discrete H1 matrix of the velocity and M_p the pressure mass
// matrix.  This is the characterization used in Huang-Liu-Wang, SIAM J. Sci.
// Comput. 35(5), B953-B986 (2013), eq. (4.2), applied cell by cell.
//
// Two caveats, both reported by the program itself:
//
//   * D is assembled here rather than taken from the solver, so it is
//     checked against the physical-space divergence of poisson() by
//     synthesis and analysis on the full grid.
//   * A is a modelling choice: the solver never forms an H1 matrix, and
//     beta_h depends on which discrete norm is used.  What does not depend
//     on it is the behaviour under refinement, O(1) versus O(h).
//
// Nothing assumes that the phase matrices collapse to scalars: D and A are
// assembled with the full phase blocks.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "blas.h"
#include "config.h"

extern "C" void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv,
                       double* b, int* ldb, int* info);

namespace {

using std::vector;

struct Geometry {
    int nr, nphi, nz;
    double r0, R, Lz;
    double dr, dphi, dz;

    double rho(int j) const { return r0+j*dr; }        // radial faces, u
    double r(int j) const { return r0+(j-0.5)*dr; }    // cell centres
};

bool endpoint(int q, int n) {
    return q == 0 || 2*q == n;
}

// Degrees of freedom of one Fourier cell.  u lives on interior faces only:
// the walls carry u = 0, and all three components vanish there.
struct BlockSpace {
    int sm, sl, ph;
    int nu, nw, nv, nU, nP;

    BlockSpace(const Geometry& g, int m, int l)
        : sm(endpoint(m, g.nphi) ? 1 : 2)
        , sl(endpoint(l, g.nz) ? 1 : 2)
        , ph(sm*sl)
        , nu((g.nr-1)*ph)
        , nw(g.nr*ph)
        , nv(g.nr*ph)
        , nU(nu+nw+nv)
        , nP(g.nr*ph)
    { }

    int iu(int j, int a) const { return (j-1)*ph+a; }
    int iw(int j, int a) const { return nu+(j-1)*ph+a; }
    int iv(int j, int a) const { return nu+nw+(j-1)*ph+a; }
    int ip(int j, int a) const { return (j-1)*ph+a; }
};

// One-dimensional shift in the real cos/sin basis: a rotation for interior
// frequencies, a sign for the endpoints.
void one_dim_shift(int q, int n, int count, int direction, double* out) {
    const double angle = direction*2*M_PI*static_cast<double>(q)/n;
    if (count == 1) {
        out[0] = std::cos(angle);
        return;
    }
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    out[0] = c;  out[1] = s;
    out[2] = -s; out[3] = c;
}

// Tensor-product shift on the packed phases; the layout matches
// NSCylFourierBlockNative: phase = phi_position*z_phases + z_position.
vector<double> shift_matrix(const Geometry& g, const BlockSpace& s,
                            int m, int l, bool azimuthal, int direction) {
    double one[4] = {0, 0, 0, 0};
    one_dim_shift(azimuthal ? m : l, azimuthal ? g.nphi : g.nz,
                  azimuthal ? s.sm : s.sl, direction, one);

    vector<double> out(static_cast<std::size_t>(s.ph)*s.ph, 0.0);
    for (int pr = 0; pr < s.sm; ++pr) {
        for (int zr = 0; zr < s.sl; ++zr) {
            const int row = pr*s.sl+zr;
            for (int pc = 0; pc < s.sm; ++pc) {
                for (int zc = 0; zc < s.sl; ++zc) {
                    const int col = pc*s.sl+zc;
                    if (azimuthal && zr == zc) {
                        out[row*s.ph+col] = one[pr*s.sm+pc];
                    } else if (!azimuthal && pr == pc) {
                        out[row*s.ph+col] = one[zr*s.sl+zc];
                    }
                }
            }
        }
    }
    return out;
}

// I - S, the one-sided backward difference symbol
vector<double> backward(const vector<double>& shift, int n) {
    vector<double> out(shift.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = -shift[i];
    }
    for (int i = 0; i < n; ++i) {
        out[i*n+i] += 1.0;
    }
    return out;
}

// a^T a
vector<double> gram(const vector<double>& a, int n) {
    vector<double> out(static_cast<std::size_t>(n)*n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += a[k*n+i]*a[k*n+j];
            }
            out[i*n+j] = sum;
        }
    }
    return out;
}

// The real packed basis: phase a = phi_position*z_phases + z_position,
// cosine at position 0 and sine at position 1, as in PeriodicPackedFFT2.
double basis(const Geometry& g, const BlockSpace& s, int m, int l,
             int a, int i, int k) {
    const int pp = a/s.sl;
    const int zp = a%s.sl;
    const double pa = 2*M_PI*static_cast<double>(m)*i/g.nphi;
    const double za = 2*M_PI*static_cast<double>(l)*k/g.nz;
    const double fp = (pp == 0) ? std::cos(pa) : std::sin(pa);
    const double fz = (zp == 0) ? std::cos(za) : std::sin(za);
    return fp*fz;
}

// --- discrete divergence of one Fourier cell, nP x nU --------------------
vector<double> assemble_divergence(const Geometry& g, const BlockSpace& s,
                                   int m, int l) {
    const auto b_phi = backward(shift_matrix(g, s, m, l, true, -1), s.ph);
    const auto b_z = backward(shift_matrix(g, s, m, l, false, -1), s.ph);

    vector<double> D(static_cast<std::size_t>(s.nP)*s.nU, 0.0);
    for (int j = 1; j <= g.nr; ++j) {
        const double rj = g.r(j);
        for (int a = 0; a < s.ph; ++a) {
            const std::size_t row = static_cast<std::size_t>(s.ip(j, a))*s.nU;
            if (j <= g.nr-1) {
                D[row+s.iu(j, a)] += g.rho(j)/(rj*g.dr);
            }
            if (j-1 >= 1) {
                D[row+s.iu(j-1, a)] -= g.rho(j-1)/(rj*g.dr);
            }
            for (int b = 0; b < s.ph; ++b) {
                D[row+s.iw(j, b)] += b_phi[a*s.ph+b]/(rj*g.dphi);
                D[row+s.iv(j, b)] += b_z[a*s.ph+b]/g.dz;
            }
        }
    }
    return D;
}

// --- discrete H1 matrix of the velocity, nU x nU -------------------------
vector<double> assemble_h1(const Geometry& g, const BlockSpace& s,
                           int m, int l) {
    const auto gram_phi = gram(
        backward(shift_matrix(g, s, m, l, true, -1), s.ph), s.ph);
    const auto gram_z = gram(
        backward(shift_matrix(g, s, m, l, false, -1), s.ph), s.ph);

    const double cell = g.dr*g.dphi*g.dz;
    vector<double> A(static_cast<std::size_t>(s.nU)*s.nU, 0.0);
    auto add = [&](int i, int j, double value) {
        A[static_cast<std::size_t>(i)*s.nU+j] += value;
    };

    for (int j = 1; j <= g.nr-1; ++j) {                 // u, on faces
        const double vol = g.rho(j)*cell;
        const double rr = g.rho(j)*g.rho(j);
        for (int a = 0; a < s.ph; ++a) {
            add(s.iu(j, a), s.iu(j, a), vol);
            for (int b = 0; b < s.ph; ++b) {
                add(s.iu(j, a), s.iu(j, b),
                    vol*gram_phi[a*s.ph+b]/(rr*g.dphi*g.dphi)
                    +vol*gram_z[a*s.ph+b]/(g.dz*g.dz));
            }
        }
    }
    for (int j = 1; j <= g.nr; ++j) {                   // w and v, in cells
        const double vol = g.r(j)*cell;
        const double rr = g.r(j)*g.r(j);
        for (int a = 0; a < s.ph; ++a) {
            add(s.iw(j, a), s.iw(j, a), vol);
            add(s.iv(j, a), s.iv(j, a), vol);
            for (int b = 0; b < s.ph; ++b) {
                const double t =
                    vol*gram_phi[a*s.ph+b]/(rr*g.dphi*g.dphi)
                    +vol*gram_z[a*s.ph+b]/(g.dz*g.dz);
                add(s.iw(j, a), s.iw(j, b), t);
                add(s.iv(j, a), s.iv(j, b), t);
            }
        }
    }

    // Radial stiffness, assembled over the gaps between neighbouring
    // positions rather than over the positions themselves: the weight
    // c = r_mid*dphi*dz/delta varies with r, so a diagonal built as 2*c_j
    // instead of c_{j-1}+c_j leaves an O(1/dr) row-sum error and destroys
    // the h-independence of the norm.
    const double tangential_area = g.dphi*g.dz;
    auto gap = [&](int i, int j, double weight) {
        if (i >= 0) { add(i, i, weight); }
        if (j >= 0) { add(j, j, weight); }
        if (i >= 0 && j >= 0) {
            add(i, j, -weight);
            add(j, i, -weight);
        }
    };

    for (int a = 0; a < s.ph; ++a) {
        // u on faces 0..nr, the walls carrying zero
        for (int j = 0; j <= g.nr-1; ++j) {
            const double weight = g.r(j+1)*tangential_area/g.dr;
            const int lo = (j >= 1) ? s.iu(j, a) : -1;
            const int hi = (j+1 <= g.nr-1) ? s.iu(j+1, a) : -1;
            gap(lo, hi, weight);
        }
        // w and v in cells 1..nr; the walls sit half a cell away
        for (int j = 1; j <= g.nr-1; ++j) {
            const double weight = g.rho(j)*tangential_area/g.dr;
            gap(s.iw(j, a), s.iw(j+1, a), weight);
            gap(s.iv(j, a), s.iv(j+1, a), weight);
        }
        const double inner = g.rho(0)*tangential_area/(0.5*g.dr);
        const double outer = g.rho(g.nr)*tangential_area/(0.5*g.dr);
        gap(s.iw(1, a), -1, inner);
        gap(s.iv(1, a), -1, inner);
        gap(s.iw(g.nr, a), -1, outer);
        gap(s.iv(g.nr, a), -1, outer);
    }
    return A;
}

// --- verification of D against the physical-space divergence -------------
//
// A random block vector is synthesized onto the full (phi,z) grid, the
// physical divergence of poisson() (src/ns_cyl.cpp:405) is applied there,
// and the result is projected back.  A wrong phase convention or index
// layout in assemble_divergence shows up here.
double verify_divergence(const Geometry& g, int m, int l, std::mt19937& rng) {
    const BlockSpace s(g, m, l);
    const auto D = assemble_divergence(g, s, m, l);

    std::uniform_real_distribution<double> pick(-1.0, 1.0);
    vector<double> U(s.nU);
    for (double& value : U) {
        value = pick(rng);
    }

    const std::size_t stride = static_cast<std::size_t>(g.nr)+1;
    vector<double> u(static_cast<std::size_t>(g.nphi)*g.nz*stride, 0.0);
    vector<double> w(u.size(), 0.0);
    vector<double> v(u.size(), 0.0);
    auto at = [&](vector<double>& f, int i, int k, int j) -> double& {
        return f[(static_cast<std::size_t>(i)*g.nz+k)*stride+j];
    };

    for (int i = 0; i < g.nphi; ++i) {
        for (int k = 0; k < g.nz; ++k) {
            for (int a = 0; a < s.ph; ++a) {
                const double b = basis(g, s, m, l, a, i, k);
                for (int j = 1; j <= g.nr-1; ++j) {
                    at(u, i, k, j) += U[s.iu(j, a)]*b;
                }
                for (int j = 1; j <= g.nr; ++j) {
                    at(w, i, k, j) += U[s.iw(j, a)]*b;
                    at(v, i, k, j) += U[s.iv(j, a)]*b;
                }
            }
        }
    }

    vector<double> div(u.size(), 0.0);
    for (int i = 0; i < g.nphi; ++i) {
        const int im = (i-1+g.nphi)%g.nphi;
        for (int k = 0; k < g.nz; ++k) {
            const int km = (k-1+g.nz)%g.nz;
            for (int j = 1; j <= g.nr; ++j) {
                const double rj = g.r(j);
                at(div, i, k, j) =
                    ((rj+0.5*g.dr)*at(u, i, k, j)
                     -(rj-0.5*g.dr)*at(u, i, k, j-1))/(rj*g.dr)
                    +(at(v, i, k, j)-at(v, i, km, j))/g.dz
                    +(at(w, i, k, j)-at(w, im, k, j))/(g.dphi*rj);
            }
        }
    }

    double error = 0;
    double scale = 0;
    for (int j = 1; j <= g.nr; ++j) {
        for (int a = 0; a < s.ph; ++a) {
            double num = 0;
            double den = 0;
            for (int i = 0; i < g.nphi; ++i) {
                for (int k = 0; k < g.nz; ++k) {
                    const double b = basis(g, s, m, l, a, i, k);
                    num += at(div, i, k, j)*b;
                    den += b*b;
                }
            }
            const double physical = num/den;

            double blockwise = 0;
            for (int c = 0; c < s.nU; ++c) {
                blockwise += D[static_cast<std::size_t>(s.ip(j, a))*s.nU+c]
                            *U[c];
            }
            error = std::max(error, std::abs(physical-blockwise));
            scale = std::max(scale, std::abs(physical));
        }
    }
    return scale > 0 ? error/scale : error;
}

struct BlockResult {
    int m = 0, l = 0, phases = 0;
    int pressure_size = 0, velocity_size = 0;
    int null_dimension = 0;
    double beta = 0;
    double imaginary = 0;
};

BlockResult block_lbb(const Geometry& g, int m, int l, double null_tolerance) {
    const BlockSpace s(g, m, l);
    const auto D = assemble_divergence(g, s, m, l);
    auto A = assemble_h1(g, s, m, l);

    // X = A^{-1} D^T; A is symmetric, so column major needs no transpose
    vector<double> X(static_cast<std::size_t>(s.nU)*s.nP, 0.0);
    for (int c = 0; c < s.nP; ++c) {
        for (int i = 0; i < s.nU; ++i) {
            X[static_cast<std::size_t>(c)*s.nU+i] =
                D[static_cast<std::size_t>(c)*s.nU+i];
        }
    }
    {
        int n = s.nU, nrhs = s.nP, lda = s.nU, ldb = s.nU, info = 0;
        vector<int> pivots(s.nU);
        dgesv_(&n, &nrhs, A.data(), &lda, pivots.data(),
               X.data(), &ldb, &info);
        if (info != 0) {
            throw std::runtime_error(
                "H1 matrix solve failed with info="+std::to_string(info));
        }
    }

    const double cell = g.dr*g.dphi*g.dz;
    vector<double> root(s.nP);
    for (int j = 1; j <= g.nr; ++j) {
        for (int a = 0; a < s.ph; ++a) {
            root[s.ip(j, a)] = std::sqrt(g.r(j)*cell);
        }
    }

    vector<double> S(static_cast<std::size_t>(s.nP)*s.nP, 0.0);
    for (int i = 0; i < s.nP; ++i) {
        for (int c = 0; c < s.nP; ++c) {
            double sum = 0;
            for (int k = 0; k < s.nU; ++k) {
                sum += D[static_cast<std::size_t>(i)*s.nU+k]
                      *X[static_cast<std::size_t>(c)*s.nU+k];
            }
            S[static_cast<std::size_t>(c)*s.nP+i] = root[i]*sum*root[c];
        }
    }

    vector<double> real(s.nP), imaginary(s.nP), dummy(1), work(8*s.nP);
    int n = s.nP, one = 1, info = 0;
    fdm::lapack::geev("N", "N", n, S.data(), n, real.data(), imaginary.data(),
                      dummy.data(), one, dummy.data(), one,
                      work.data(), static_cast<int>(work.size()), &info);
    if (info != 0) {
        throw std::runtime_error("geev failed with info="+std::to_string(info));
    }

    BlockResult result;
    result.m = m;
    result.l = l;
    result.phases = s.ph;
    result.pressure_size = s.nP;
    result.velocity_size = s.nU;

    double largest = 0;
    for (int i = 0; i < s.nP; ++i) {
        largest = std::max(largest, std::abs(real[i]));
        result.imaginary = std::max(result.imaginary, std::abs(imaginary[i]));
    }
    const double cut = null_tolerance*std::max(largest, 1e-300);

    double smallest = std::numeric_limits<double>::infinity();
    for (int i = 0; i < s.nP; ++i) {
        if (real[i] <= cut) {
            ++result.null_dimension;
        } else {
            smallest = std::min(smallest, real[i]);
        }
    }
    result.beta = std::isfinite(smallest) ? std::sqrt(smallest) : 0.0;
    return result;
}

} // namespace

int main(int argc, char** argv) {
    std::string config_file;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-c" && i+1 < argc) {
            config_file = argv[++i];
        }
    }

    Config config;
    if (!config_file.empty()) {
        config.open(config_file);
    }
    config.rewrite(argc, argv);

    Geometry g;
    g.nr = config.get("ns", "nr", 32);
    g.nphi = config.get("ns", "nphi", 32);
    g.nz = config.get("ns", "nz", 32);
    g.r0 = config.get("ns", "r", M_PI/2);
    g.R = config.get("ns", "R", M_PI);
    g.Lz = config.get("ns", "h2", 10.0)-config.get("ns", "h1", 0.0);
    g.dr = (g.R-g.r0)/g.nr;
    g.dphi = 2*M_PI/g.nphi;
    g.dz = g.Lz/g.nz;

    const double null_tolerance = config.get("lbb", "null_tol", 1e-10);
    const int verbose = config.get("lbb", "verbose", 0);
    const int verify = config.get("lbb", "verify", 1);
    const double verify_tolerance = config.get("lbb", "verify_tol", 1e-12);

    printf("NSCyl discrete inf-sup (LBB) test\n");
    printf("grid: nr=%d nphi=%d nz=%d  r0=%.9g R=%.9g Lz=%.9g\n",
           g.nr, g.nphi, g.nz, g.r0, g.R, g.Lz);
    printf("eta=r0/R=%.6f  cell dr:r0*dphi:dz = %.4g:%.4g:%.4g\n",
           g.r0/g.R, g.dr, g.r0*g.dphi, g.dz);
    fflush(stdout);

    int failures = 0;

    if (verify) {
        printf("\nverifying D against the physical divergence of poisson()\n");
        std::mt19937 rng(20260907);
        double worst = 0;
        int worst_m = 0, worst_l = 0;
        const int mm = std::min(4, g.nphi/2);
        const int ll = std::min(4, g.nz/2);
        for (int m = 0; m <= mm; ++m) {
            for (int l = 0; l <= ll; ++l) {
                const double e = verify_divergence(g, m, l, rng);
                if (e > worst) {
                    worst = e;
                    worst_m = m;
                    worst_l = l;
                }
            }
        }
        printf("  worst relative error %.3e in block (%d,%d)  [%s]\n",
               worst, worst_m, worst_l,
               worst < verify_tolerance ? "PASS" : "FAIL");
        if (!(worst < verify_tolerance)) {
            ++failures;
        }
    }

    printf("\nsweeping %d blocks\n", (g.nphi/2+1)*(g.nz/2+1));
    fflush(stdout);

    const int lm = g.nphi/2+1;
    const int ln = g.nz/2+1;
    vector<BlockResult> results(static_cast<std::size_t>(lm)*ln);

#pragma omp parallel for schedule(dynamic)
    for (int index = 0; index < lm*ln; ++index) {
        results[index] = block_lbb(g, index/ln, index%ln, null_tolerance);
    }

    BlockResult worst;
    worst.beta = std::numeric_limits<double>::infinity();
    int total_null = 0;
    double worst_imaginary = 0;

    for (const BlockResult& r : results) {
        total_null += r.null_dimension;
        worst_imaginary = std::max(worst_imaginary, r.imaginary);
        if (verbose) {
            printf("  block (%2d,%2d) phases=%d nP=%3d nU=%4d "
                   "null=%d beta=%.9e\n",
                   r.m, r.l, r.phases, r.pressure_size, r.velocity_size,
                   r.null_dimension, r.beta);
        }
        const int expected = (r.m == 0 && r.l == 0) ? 1 : 0;
        if (r.null_dimension != expected) {
            printf("  WARNING block (%d,%d): null dimension %d, "
                   "expected %d\n", r.m, r.l, r.null_dimension, expected);
        }
        if (r.beta < worst.beta) {
            worst = r;
        }
    }

    printf("\nnull space total %d (expected 1: the constant in block (0,0))\n",
           total_null);
    printf("largest |Im| among eigenvalues %.3e (round-off expected)\n",
           worst_imaginary);
    printf("beta_h = %.9e   attained in block (%d,%d)\n",
           worst.beta, worst.m, worst.l);
    if (total_null != 1) {
        ++failures;
    }
    printf("\n%s\n", failures == 0 ? "PASSED" : "FAILED");
    printf("note: beta_h depends on the choice of discrete H1 norm; the "
           "meaningful\n      quantity is its behaviour under refinement.\n");
    return failures == 0 ? 0 : 1;
}
