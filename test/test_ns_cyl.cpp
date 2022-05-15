#include <string>
#include <vector>
#include <climits>
#include <cmath>

#include "tensor.h"
#include "config.h"

using namespace std;

template<typename T, bool check>
class NSCyl {
public:
    using tensor = fdm::tensor<T,3,check>;
    using matrix = fdm::tensor<T,2,check>;

    const double Re;
    const double dt;

    NSCyl(const Config& c)
        : Re(c.get("ns", "Re", 1.0))
        , dt(c.get("ns", "dt", 0.001))
    { }

    void step() { }
    void plot() { }
};

template<typename T, bool check>
void calc(const Config& c) {
    NSCyl<T, true> ns(c);

    const int steps = c.get("ns", "steps", 1);
    const int plot_interval = c.get("plot", "interval", 100);
    int i;

    ns.plot();
    for (i = 0; i < steps; i++) {
        ns.step();

        if ((i+1) % plot_interval == 0) {
            ns.plot();
        }
    }
}

// Флетчер, том 2, страница 398
int main(int argc, char** argv) {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);
    c.rewrite(argc, argv);

    bool check = c.get("other", "check", 0) == 1;
    if (check) {
        calc<double,true>(c);
    } else {
        calc<double,false>(c);
    }

    return 0;
}
