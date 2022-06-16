#pragma once

#include <vector>
#include "config.h"

namespace fdm {

class eigenvectors_storage {
public:
    eigenvectors_storage(const std::string& filename)
        : filename(filename)
    { }

    // TODO: split vector into components (u,v,w,p...)
    void save(const std::vector<std::vector<float>>& eigenvectors,
              const std::vector<int>& indices, const Config& config);
    void save(const std::vector<std::vector<double>>& eigenvectors,
              const std::vector<int>& indices, const Config& config);

    void load(std::vector<std::vector<float>>& eigenvectors, Config& config);
    void load(std::vector<std::vector<double>>& eigenvectors, Config& config);

private:
    std::string filename;

    template<typename T>
    void save_(const std::vector<std::vector<T>>& eigenvectors,
               const std::vector<int>& indices, const Config& config);
    template<typename T>
    void load_(std::vector<std::vector<T>>& eigenvectors, Config& config);
};


} // namespace fdm
