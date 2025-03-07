#include <netcdf.h>
#include <string.h>
#include <stdexcept>

#include "asp_misc.h"
#include "verify.h"
#include "eigenvectors_storage.h"

using namespace asp;
using std::vector;
using std::is_same;

#ifdef _WIN32
static FILE* fmemopen(void* ptr, size_t size, const char* mode) {
    throw std::runtime_error("Unimplemented");
}
#endif

#define nc_call(expr) do {                        \
        int code = expr;                          \
        verify(code == 0, nc_strerror(code));     \
    } while(0);

namespace fdm {

template<typename T>
void eigenvectors_storage::save_(const vector<vector<T>>& eigenvectors,
                                 const std::vector<int>& indices,
                                 const Config& c)
{
    verify(eigenvectors.size() > 0);
    verify(indices.size() > 0);
    verify(eigenvectors.size() >= indices.size());

    int N = eigenvectors[0].size();
    int count = indices.size();

    for (auto& i : indices) {
        verify(i < static_cast<int>(eigenvectors.size()));
    }

    for (auto& vec : eigenvectors) {
        verify(static_cast<int>(vec.size()) == N);
    }

    vector<char> mem(1024000);
    FILE* cf = fmemopen(&mem[0], mem.size(), "wb");
    c.print(cf);
    fclose(cf);

    int ncid;
    nc_call(nc_create(filename.c_str(), NC_CLOBBER, &ncid));
    int N_dim;

    nc_call(nc_put_att_text(ncid, NC_GLOBAL, "config", strlen(&mem[0]), &mem[0]));
    nc_call(nc_def_dim(ncid, "N", N, &N_dim));

    int type = NC_DOUBLE;
    if constexpr(is_same<float,T>::value) {
        type = NC_FLOAT;
    }

    vector<int> uids;
    for (int  i = 0; i < count; i++) {
        int udims[] = {N_dim}; int uid;
        nc_call(nc_def_var(ncid, format("vec_%d", i).c_str(), type, 1, udims, &uid));
        uids.push_back(uid);
    }

    nc_call(nc_enddef(ncid));

    for (int  i = 0; i < count; i++) {
        int j = indices[i];
        int off = 0;
        if constexpr(is_same<float,T>::value) {
            nc_call(nc_put_var_float(ncid, uids[i], &eigenvectors[j][off]));
        } else {
            nc_call(nc_put_var_double(ncid, uids[i], &eigenvectors[j][off]));
        }
    }


    nc_call(nc_close(ncid));
}

void eigenvectors_storage::save(const std::vector<std::vector<float>>& eigenvectors,
                                const std::vector<int>& indices,
                                const Config& config)
{
    save_(eigenvectors, indices, config);
}

void eigenvectors_storage::save(const std::vector<std::vector<double>>& eigenvectors,
                                const std::vector<int>& indices,
                                const Config& config)
{
    save_(eigenvectors, indices, config);
}

template<typename T>
void eigenvectors_storage::load_(std::vector<std::vector<T>>& eigenvectors, Config& config)
{
    int ncid;
    nc_call(nc_open(filename.c_str(), NC_NOWRITE, &ncid));
    size_t len;
    nc_call(nc_inq_attlen(ncid, NC_GLOBAL, "config", &len));
    vector<char> mem(len+1, 0);
    nc_call(nc_get_att_text(ncid, NC_GLOBAL, "config", &mem[0]));
    FILE* cf = fmemopen(&mem[0], mem.size(), "rb");
    config.load(cf);
    fclose(cf);

    int ndims, nvars, natts, unlimdimid;
    nc_call(nc_inq(ncid, &ndims, &nvars, &natts, &unlimdimid));
    eigenvectors.resize(nvars);
    for (int i = 0; i < nvars; i++) {
        int varid, dimid;
        size_t len;
        nc_call(nc_inq_varid(ncid, format("vec_%d", i).c_str(), &varid));
        nc_call(nc_inq_vardimid(ncid, varid, &dimid));
        nc_call(nc_inq_dimlen(ncid, dimid, &len));
        eigenvectors[i].resize(len);
        nc_call(nc_get_var(ncid, varid, &eigenvectors[i][0]));
    }

    nc_call(nc_close(ncid));
}

void eigenvectors_storage::load(std::vector<std::vector<float>>& eigenvectors, Config& config)
{
    load_(eigenvectors, config);
}

void eigenvectors_storage::load(std::vector<std::vector<double>>& eigenvectors, Config& config)
{
    load_(eigenvectors, config);
}

} // namespace fdm
