#include "unixbench_score.h"

#include <algorithm>
#include <cmath>

namespace fdm {

double unixbench_score(std::vector<double>& data) {
    if (data.empty()) return 0.0;
    std::sort(data.begin(), data.end(), std::greater<double>());
    int keep = (2 * data.size()) / 3;
    data.resize(keep);
    double score = 0.0;
    for (auto d : data) {
        score += log(d);
    }
    score = exp(score / keep);
    return score;
}

} // namespace fdm