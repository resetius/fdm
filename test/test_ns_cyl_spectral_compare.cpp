#include <netcdf.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "ns_cyl_spectral_compare.h"
#include "ns_cyl_spectral_storage.h"

namespace {

void nc_check(int code, const std::string& operation) {
    if (code != NC_NOERR) {
        throw std::runtime_error(operation+": "+nc_strerror(code));
    }
}

std::string scalar_type(const std::string& filename) {
    int ncid = -1;
    nc_check(nc_open(filename.c_str(), NC_NOWRITE, &ncid),
             "opening "+filename);
    try {
        std::size_t size = 0;
        nc_check(nc_inq_attlen(ncid, NC_GLOBAL, "scalar_type", &size),
                 "reading scalar_type length");
        std::string result(size, '\0');
        nc_check(nc_get_att_text(ncid, NC_GLOBAL, "scalar_type",
                                 result.data()),
                 "reading scalar_type");
        nc_check(nc_close(ncid), "closing "+filename);
        return result;
    } catch (...) {
        nc_close(ncid);
        throw;
    }
}

double maximum_finite(const fdm::NSCylSpectralComparison& comparison,
                      double fdm::NSCylSpectralBlockComparison::*member) {
    double result = std::numeric_limits<double>::quiet_NaN();
    for (const auto& block : comparison.blocks) {
        const double value = block.*member;
        if (std::isfinite(value)) {
            result = std::isfinite(result) ? std::max(result, value) : value;
        }
    }
    return result;
}

void write_csv(const std::string& filename,
               const fdm::NSCylSpectralComparison& comparison) {
    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("cannot create comparison CSV: "+filename);
    }
    output << "m,l,status,coarse_phases,fine_phases,coarse_groups,"
              "fine_groups,coarse_dimension,fine_dimension,"
              "coarse_leading_growth,fine_leading_growth,"
              "max_growth_change,max_frequency_change,"
              "right_subspace_sine,right_velocity_subspace_sine,"
              "left_subspace_sine\n";
    output << std::setprecision(17);
    for (const auto& block : comparison.blocks) {
        output << block.m << ',' << block.l << ','
               << fdm::ns_cyl_spectral_comparison_status_name(block.status)
               << ',' << block.coarse_phase_count
               << ',' << block.fine_phase_count
               << ',' << block.coarse_group_count
               << ',' << block.fine_group_count
               << ',' << block.coarse_dimension
               << ',' << block.fine_dimension
               << ',' << block.coarse_leading_growth
               << ',' << block.fine_leading_growth
               << ',' << block.max_growth_change
               << ',' << block.max_frequency_change
               << ',' << block.right_subspace_sine
               << ',' << block.right_velocity_subspace_sine
               << ',' << block.left_subspace_sine << '\n';
    }
}

template<typename T>
int run(const std::string& coarse_filename,
        const std::string& fine_filename,
        const std::string& csv_filename) {
    fdm::NSCylSpectralModeSet<T> coarse_modes;
    fdm::NSCylSpectralModeSet<T> fine_modes;
    fdm::NSCylSpectralMetadata coarse_metadata;
    fdm::NSCylSpectralMetadata fine_metadata;
    fdm::NSCylSpectralStorage(coarse_filename).load(
        coarse_modes, coarse_metadata);
    fdm::NSCylSpectralStorage(fine_filename).load(
        fine_modes, fine_metadata);

    const auto comparison = fdm::compare_ns_cyl_spectral_mode_sets(
        coarse_modes, coarse_metadata, fine_modes, fine_metadata);
    std::cout << "coarse: " << coarse_metadata.nr << 'x'
              << coarse_metadata.nphi << 'x' << coarse_metadata.nz
              << "  groups=" << coarse_modes.size()
              << "  real_dimension=" << coarse_modes.real_dimension()
              << "  dt=" << coarse_metadata.dt
              << "  L_steps=" << coarse_metadata.operator_steps << '\n';
    std::cout << "fine:   " << fine_metadata.nr << 'x'
              << fine_metadata.nphi << 'x' << fine_metadata.nz
              << "  groups=" << fine_modes.size()
              << "  real_dimension=" << fine_modes.real_dimension()
              << "  dt=" << fine_metadata.dt
              << "  L_steps=" << fine_metadata.operator_steps << '\n';
    if (coarse_metadata.dt != fine_metadata.dt) {
        std::cout << "warning: dt differs; spatial and temporal changes are "
                     "mixed in eigenvectors\n";
    }
    if (coarse_metadata.operator_steps != fine_metadata.operator_steps) {
        std::cout << "note: L_steps differs; growth/frequency are normalized "
                     "by physical time\n";
    }

    std::cout << "\n"
              << std::setw(3) << "m" << ' '
              << std::setw(3) << "l" << ' '
              << std::setw(25) << "status" << ' '
              << std::setw(7) << "phase" << ' '
              << std::setw(7) << "groups" << ' '
              << std::setw(7) << "dim" << ' '
              << std::setw(11) << "d_growth" << ' '
              << std::setw(11) << "d_freq" << ' '
              << std::setw(11) << "sin_R" << ' '
              << std::setw(11) << "sin_R_vel" << ' '
              << std::setw(11) << "sin_L" << '\n';
    std::cout << std::scientific << std::setprecision(3);
    for (const auto& block : comparison.blocks) {
        std::cout << std::setw(3) << block.m << ' '
                  << std::setw(3) << block.l << ' '
                  << std::setw(25)
                  << fdm::ns_cyl_spectral_comparison_status_name(block.status)
                  << ' ' << block.coarse_phase_count << '/'
                  << block.fine_phase_count << "     "
                  << block.coarse_group_count << '/'
                  << block.fine_group_count << "     "
                  << block.coarse_dimension << '/'
                  << block.fine_dimension << "     "
                  << std::setw(11) << block.max_growth_change << ' '
                  << std::setw(11) << block.max_frequency_change << ' '
                  << std::setw(11) << block.right_subspace_sine << ' '
                  << std::setw(11) << block.right_velocity_subspace_sine << ' '
                  << std::setw(11) << block.left_subspace_sine << '\n';
    }

    std::cout << "\nblocks: common=" << comparison.common_blocks
              << " comparable=" << comparison.comparable_blocks
              << " phase_layout_change=" << comparison.phase_layout_changes
              << " unstable_dimension_change="
              << comparison.unstable_dimension_changes
              << " coarse_only=" << comparison.coarse_only_blocks
              << " fine_only=" << comparison.fine_only_blocks << '\n';
    std::cout << "max over matched blocks: |d_growth|="
              << maximum_finite(comparison,
                    &fdm::NSCylSpectralBlockComparison::max_growth_change)
              << " |d_frequency|="
              << maximum_finite(comparison,
                    &fdm::NSCylSpectralBlockComparison::max_frequency_change)
              << " sin_R="
              << maximum_finite(comparison,
                    &fdm::NSCylSpectralBlockComparison::right_subspace_sine)
              << " sin_R_velocity="
              << maximum_finite(comparison,
                    &fdm::NSCylSpectralBlockComparison::right_velocity_subspace_sine)
              << " sin_L="
              << maximum_finite(comparison,
                    &fdm::NSCylSpectralBlockComparison::left_subspace_sine)
              << '\n';
    std::cout << "sin_R_velocity uses cylindrical radial weights; sin_L is "
                 "an algebraic comparison of the discrete Euclidean-adjoint "
                 "coordinates.\n";

    if (!csv_filename.empty()) {
        write_csv(csv_filename, comparison);
        std::cout << "CSV: " << csv_filename << '\n';
    }
    return 0;
}

void usage(const char* program) {
    std::cerr << "usage: " << program
              << " [--csv output.csv] coarse_spectrum.nc fine_spectrum.nc\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string csv_filename;
        int argument = 1;
        if (argument < argc && std::string(argv[argument]) == "--csv") {
            if (++argument >= argc) {
                usage(argv[0]);
                return 2;
            }
            csv_filename = argv[argument++];
        }
        if (argc-argument != 2) {
            usage(argv[0]);
            return 2;
        }
        const std::string coarse_filename = argv[argument++];
        const std::string fine_filename = argv[argument];
        const std::string coarse_type = scalar_type(coarse_filename);
        const std::string fine_type = scalar_type(fine_filename);
        if (coarse_type != fine_type) {
            throw std::runtime_error(
                "spectral scalar types differ: "+coarse_type+" vs "+fine_type);
        }
        if (coarse_type == "float32") {
            return run<float>(coarse_filename, fine_filename, csv_filename);
        }
        if (coarse_type == "float64") {
            return run<double>(coarse_filename, fine_filename, csv_filename);
        }
        throw std::runtime_error("unsupported scalar_type: "+coarse_type);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
