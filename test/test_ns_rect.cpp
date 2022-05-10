#include <string>
#include <vector>

#include "config.h"

using namespace std;


// Флетчер, том 2, страница 398
int main() {
    string config_fn = "ns_rect.ini";

    Config c;

    c.open(config_fn);

    double x1 = c.get("ns", "x1", -1.0);
    double y1 = c.get("ns", "y1", -1.0);

    double x2 = c.get("ns", "x2", 1.0);
    double y2 = c.get("ns", "y2", 1.0);

    int nx = c.get("ns", "nx", 32);
    int ny = c.get("ns", "ny", 32);

    double Re = c.get("ns", "Re", 1.0);
    double dt = c.get("ns", "dt", 0.001);

    double dx = (x2-x1)/nx;
    double dy = (y2-y1)/ny;

    int np = nx*ny; // количество внутренних точек давления (p)
    int nu = (nx-1)*ny; // количество внутренних точек скорости (u)
    int nv = nx*(ny-1); // количество внутренних точек скорости (v)
    vector<double> p, pnext, u, unext, v, vnext;
    p.resize(np); pnext.resize(np);
    u.resize(nu); unext.resize(nu);
    v.resize(nv); vnext.resize(nv);

    vector<double> G,F;
    F.resize(nu); G.resize(nv);

    /*

          v=0,u=0
         ________
     p=0 |      | u/n=0
     u=1 |      | v/n=0
         |      |
         ________
          v=0,u=0

         \int Lapl p = 0 ?

     */

    // F
    for (int i = 0; i < nx-1; i++) {
        for (int j = 0; j < ny; j++) {
            //u[];
        }
    }

    return 0;
}
