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
//   * D and the real phase layout come from NSCylFourierBlockNative.  An
//     independent physical-space synthesis check guards that shared code.
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
#include "ns_cyl_fourier_native.h"

extern "C" void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv,
                       double* b, int* ldb, int* info);

namespace {

using std::vector;
using NativeBlock = fdm::NSCylFourierBlockNative<double>;
using Component = NativeBlock::Component;

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

// --- discrete H1 matrix of the velocity, nU x nU -------------------------
vector<double> assemble_h1(const NativeBlock& block) {
    const int phases = block.phase_count();
    const int velocity_size = block.velocity_block_size();
    const auto gram_phi = gram(block.backward_phi_matrix(), phases);
    const auto gram_z = gram(block.backward_z_matrix(), phases);

    const double cell = block.dr*block.dphi*block.dz;
    vector<double> A(
        static_cast<std::size_t>(velocity_size)*velocity_size, 0.0);
    auto add = [&](int i, int j, double value) {
        A[static_cast<std::size_t>(i)*velocity_size+j] += value;
    };

    for (int j = 1; j < block.nr; ++j) {
        const double radius = block.r0+j*block.dr;
        const double volume = radius*cell;
        for (int phase = 0; phase < phases; ++phase) {
            const int row = block.velocity_block_index(
                Component::u, phase, j);
            add(row, row, volume);
            for (int column_phase = 0;
                 column_phase < phases; ++column_phase) {
                add(row, block.velocity_block_index(
                        Component::u, column_phase, j),
                    volume*gram_phi[phase*phases+column_phase]
                        /(radius*radius*block.dphi*block.dphi)
                    +volume*gram_z[phase*phases+column_phase]
                        /(block.dz*block.dz));
            }
        }
    }
    for (int j = 1; j <= block.nr; ++j) {
        const double radius = block.r0+(j-0.5)*block.dr;
        const double volume = radius*cell;
        for (int phase = 0; phase < phases; ++phase) {
            for (Component component : {Component::v, Component::w}) {
                const int row = block.velocity_block_index(
                    component, phase, j);
                add(row, row, volume);
            }
            for (int column_phase = 0;
                 column_phase < phases; ++column_phase) {
                const double t =
                    volume*gram_phi[phase*phases+column_phase]
                        /(radius*radius*block.dphi*block.dphi)
                    +volume*gram_z[phase*phases+column_phase]
                        /(block.dz*block.dz);
                for (Component component : {Component::v, Component::w}) {
                    add(block.velocity_block_index(component, phase, j),
                        block.velocity_block_index(
                            component, column_phase, j), t);
                }
            }
        }
    }

    // Radial stiffness, assembled over the gaps between neighbouring
    // positions rather than over the positions themselves: the weight
    // c = r_mid*dphi*dz/delta varies with r, so a diagonal built as 2*c_j
    // instead of c_{j-1}+c_j leaves an O(1/dr) row-sum error and destroys
    // the h-independence of the norm.
    const double tangential_area = block.dphi*block.dz;
    auto gap = [&](int i, int j, double weight) {
        if (i >= 0) { add(i, i, weight); }
        if (j >= 0) { add(j, j, weight); }
        if (i >= 0 && j >= 0) {
            add(i, j, -weight);
            add(j, i, -weight);
        }
    };

    for (int phase = 0; phase < phases; ++phase) {
        for (int j = 0; j < block.nr; ++j) {
            const double radius = block.r0+(j+0.5)*block.dr;
            const double weight = radius*tangential_area/block.dr;
            const int lo = j >= 1 ? block.velocity_block_index(
                Component::u, phase, j) : -1;
            const int hi = j+1 < block.nr ? block.velocity_block_index(
                Component::u, phase, j+1) : -1;
            gap(lo, hi, weight);
        }
        for (int j = 1; j < block.nr; ++j) {
            const double radius = block.r0+j*block.dr;
            const double weight = radius*tangential_area/block.dr;
            for (Component component : {Component::v, Component::w}) {
                gap(block.velocity_block_index(component, phase, j),
                    block.velocity_block_index(component, phase, j+1),
                    weight);
            }
        }
        const double inner =
            block.r0*tangential_area/(0.5*block.dr);
        const double outer =
            block.R*tangential_area/(0.5*block.dr);
        for (Component component : {Component::v, Component::w}) {
            gap(block.velocity_block_index(component, phase, 1), -1, inner);
            gap(block.velocity_block_index(
                component, phase, block.nr), -1, outer);
        }
    }
    return A;
}

// --- verification of native D against physical-space divergence -----------
//
// A random block vector is synthesized onto the full (phi,z) grid, the
// physical divergence of poisson() (src/ns_cyl.cpp:405) is applied there,
// and the result is projected back.  A wrong phase convention or index
// layout in NSCylFourierBlockNative shows up here.
double verify_divergence(const Config& config, int m, int l,
                         std::mt19937& rng) {
    const NativeBlock block(config, m, l);
    const int phases = block.phase_count();
    const int velocity_size = block.velocity_block_size();
    const auto D = block.velocity_divergence_matrix();

    std::uniform_real_distribution<double> pick(-1.0, 1.0);
    vector<double> U(velocity_size);
    for (double& value : U) {
        value = pick(rng);
    }

    const std::size_t stride = static_cast<std::size_t>(block.nr)+1;
    vector<double> u(
        static_cast<std::size_t>(block.nphi)*block.nz*stride, 0.0);
    vector<double> w(u.size(), 0.0);
    vector<double> v(u.size(), 0.0);
    auto at = [&](vector<double>& f, int i, int k, int j) -> double& {
        return f[(static_cast<std::size_t>(i)*block.nz+k)*stride+j];
    };

    for (int i = 0; i < block.nphi; ++i) {
        for (int k = 0; k < block.nz; ++k) {
            for (int phase = 0; phase < phases; ++phase) {
                const double basis = block.phase_value(phase, i, k);
                for (int j = 1; j < block.nr; ++j) {
                    at(u, i, k, j) += U[block.velocity_block_index(
                        Component::u, phase, j)]*basis;
                }
                for (int j = 1; j <= block.nr; ++j) {
                    at(v, i, k, j) += U[block.velocity_block_index(
                        Component::v, phase, j)]*basis;
                    at(w, i, k, j) += U[block.velocity_block_index(
                        Component::w, phase, j)]*basis;
                }
            }
        }
    }

    vector<double> div(u.size(), 0.0);
    for (int i = 0; i < block.nphi; ++i) {
        const int im = (i-1+block.nphi)%block.nphi;
        for (int k = 0; k < block.nz; ++k) {
            const int km = (k-1+block.nz)%block.nz;
            for (int j = 1; j <= block.nr; ++j) {
                const double radius = block.r0+(j-0.5)*block.dr;
                at(div, i, k, j) =
                    ((radius+0.5*block.dr)*at(u, i, k, j)
                     -(radius-0.5*block.dr)*at(u, i, k, j-1))
                        /(radius*block.dr)
                    +(at(v, i, k, j)-at(v, i, km, j))/block.dz
                    +(at(w, i, k, j)-at(w, im, k, j))
                        /(block.dphi*radius);
            }
        }
    }

    double error = 0;
    double scale = 0;
    for (int j = 1; j <= block.nr; ++j) {
        for (int phase = 0; phase < phases; ++phase) {
            double num = 0;
            double den = 0;
            for (int i = 0; i < block.nphi; ++i) {
                for (int k = 0; k < block.nz; ++k) {
                    const double basis = block.phase_value(phase, i, k);
                    num += at(div, i, k, j)*basis;
                    den += basis*basis;
                }
            }
            const double physical = num/den;

            double blockwise = 0;
            const int row = block.pressure_block_index(phase, j);
            for (int column = 0; column < velocity_size; ++column) {
                blockwise += D[static_cast<std::size_t>(row)*velocity_size
                               +column]*U[column];
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

BlockResult block_lbb(const Config& config, int m, int l,
                      double null_tolerance) {
    const NativeBlock block(config, m, l);
    const int phases = block.phase_count();
    const int velocity_size = block.velocity_block_size();
    const int pressure_size = block.pressure_block_size();
    const auto D = block.velocity_divergence_matrix();
    auto A = assemble_h1(block);

    // X = A^{-1} D^T; A is symmetric, so column major needs no transpose
    vector<double> X(
        static_cast<std::size_t>(velocity_size)*pressure_size, 0.0);
    for (int column = 0; column < pressure_size; ++column) {
        for (int row = 0; row < velocity_size; ++row) {
            X[static_cast<std::size_t>(column)*velocity_size+row] =
                D[static_cast<std::size_t>(column)*velocity_size+row];
        }
    }
    {
        int n = velocity_size;
        int nrhs = pressure_size;
        int lda = velocity_size;
        int ldb = velocity_size;
        int info = 0;
        vector<int> pivots(velocity_size);
        dgesv_(&n, &nrhs, A.data(), &lda, pivots.data(),
               X.data(), &ldb, &info);
        if (info != 0) {
            throw std::runtime_error(
                "H1 matrix solve failed with info="+std::to_string(info));
        }
    }

    const double cell = block.dr*block.dphi*block.dz;
    vector<double> root(pressure_size);
    for (int j = 1; j <= block.nr; ++j) {
        const double radius = block.r0+(j-0.5)*block.dr;
        for (int phase = 0; phase < phases; ++phase) {
            root[block.pressure_block_index(phase, j)] =
                std::sqrt(radius*cell);
        }
    }

    vector<double> S(
        static_cast<std::size_t>(pressure_size)*pressure_size, 0.0);
    for (int i = 0; i < pressure_size; ++i) {
        for (int c = 0; c < pressure_size; ++c) {
            double sum = 0;
            for (int k = 0; k < velocity_size; ++k) {
                sum += D[static_cast<std::size_t>(i)*velocity_size+k]
                      *X[static_cast<std::size_t>(c)*velocity_size+k];
            }
            S[static_cast<std::size_t>(c)*pressure_size+i] =
                root[i]*sum*root[c];
        }
    }

    vector<double> real(pressure_size), imaginary(pressure_size), dummy(1);
    vector<double> work(8*pressure_size);
    int n = pressure_size, one = 1, info = 0;
    fdm::lapack::geev("N", "N", n, S.data(), n, real.data(), imaginary.data(),
                      dummy.data(), one, dummy.data(), one,
                      work.data(), static_cast<int>(work.size()), &info);
    if (info != 0) {
        throw std::runtime_error("geev failed with info="+std::to_string(info));
    }

    BlockResult result;
    result.m = m;
    result.l = l;
    result.phases = phases;
    result.pressure_size = pressure_size;
    result.velocity_size = velocity_size;

    double largest = 0;
    for (int i = 0; i < pressure_size; ++i) {
        largest = std::max(largest, std::abs(real[i]));
        result.imaginary = std::max(result.imaginary, std::abs(imaginary[i]));
    }
    const double cut = null_tolerance*std::max(largest, 1e-300);

    double smallest = std::numeric_limits<double>::infinity();
    for (int i = 0; i < pressure_size; ++i) {
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

    const NativeBlock geometry(config, 0, 0);
    const double axial_length = geometry.h2-geometry.h1;

    const double null_tolerance = config.get("lbb", "null_tol", 1e-10);
    const int verbose = config.get("lbb", "verbose", 0);
    const int verify = config.get("lbb", "verify", 1);
    const double verify_tolerance = config.get("lbb", "verify_tol", 1e-12);

    printf("NSCyl discrete inf-sup (LBB) test\n");
    printf("grid: nr=%d nphi=%d nz=%d  r0=%.9g R=%.9g Lz=%.9g\n",
           geometry.nr, geometry.nphi, geometry.nz,
           geometry.r0, geometry.R, axial_length);
    printf("eta=r0/R=%.6f  cell dr:r0*dphi:dz = %.4g:%.4g:%.4g\n",
           geometry.r0/geometry.R, geometry.dr,
           geometry.r0*geometry.dphi, geometry.dz);
    fflush(stdout);

    int failures = 0;

    if (verify) {
        printf("\nverifying D against the physical divergence of poisson()\n");
        std::mt19937 rng(20260907);
        double worst = 0;
        int worst_m = 0, worst_l = 0;
        const int mm = std::min(4, geometry.nphi/2);
        const int ll = std::min(4, geometry.nz/2);
        for (int m = 0; m <= mm; ++m) {
            for (int l = 0; l <= ll; ++l) {
                const double e = verify_divergence(config, m, l, rng);
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

    printf("\nsweeping %d blocks\n",
           (geometry.nphi/2+1)*(geometry.nz/2+1));
    fflush(stdout);

    const int lm = geometry.nphi/2+1;
    const int ln = geometry.nz/2+1;
    vector<BlockResult> results(static_cast<std::size_t>(lm)*ln);

#pragma omp parallel for schedule(dynamic)
    for (int index = 0; index < lm*ln; ++index) {
        results[index] = block_lbb(
            config, index/ln, index%ln, null_tolerance);
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
