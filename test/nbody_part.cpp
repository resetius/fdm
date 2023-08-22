// test parts of P3M algorithm
#include <atomic>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>

#include <vector>

#include "matrix_plot.h"

#include "verify.h"

using uint = uint32_t;

union Pos {
    float r[4];
    struct {
        float x;
        float y;
        float z;
        float w;
    } v;
};

struct Index {
    int x;
    int y;
    int z;

    bool operator == (const Index& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
};

const int max_per_cell = 512;

struct Cell {
    int n;
    int data[max_per_cell];
};

struct Flags {
    int nf = 0;
};

const int nn = 32; // 16; // 32;
const int threads = nn*nn;

struct Test {
    int n;
    std::vector<Pos> pos;
    std::vector<Pos> vel;
    std::vector<Pos> F;
    std::vector<Flags> Fl;
    std::vector<int> list;
    std::atomic_int list_heads[nn][nn];
    Cell cells[nn][nn][nn];

    float l;
    float dt;
    Pos origin;
};

static uint32_t rand_(uint32_t* seed) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    return *seed;
}

static uint32_t rand_max = UINT_MAX;

void gen(Test* t) {
    uint32_t seed = 10;
    for (int i = 0; i < t->n; i++) {
        Pos b;
        for (int j = 0; j < 3; j++) {
            b.r[j] = t->l * (double)rand_(&seed) / (double)rand_max - t->l/2;
        }
        b.r[3] = 0.2 + 1.5*(double)rand_(&seed) / (double)rand_max;
        t->pos.emplace_back(b);

        Pos v; v.r[0] = v.r[1] = v.r[2] = v.r[3] = 0;
        t->vel.emplace_back(v);
        t->F.emplace_back(v);
        t->Fl.emplace_back();
    }
    verify((int)t->pos.size() == t->n);
}

void parts_thread_init(Test* t, int threads, int globalIndex, int x, int y) {
    auto particles = t->n;
    uint work_size = (particles + threads - 1) / threads;
    uint from = globalIndex * work_size;
    uint to = std::min<uint>(particles, from+work_size);

    for (uint i = from; i < to; i++) {
        t->list[i] = -1;
    }

    t->list_heads[y][x] = -1;
}

uint idx(float x, float h) {
    return (uint) floor((fabsf(x)*0.9999999) / h);
}

void parts_thread(Test* t, int threads, int globalIndex, int x, int y) {
    float l = t->l;
    auto particles = t->n;
    uint work_size = (particles + threads - 1) / threads;
    uint from = globalIndex * work_size;
    uint to = std::min<uint>(particles, from+work_size);

    float h32 = l/nn;
    // lists by x,y
    for (uint i = from; i < to; i++) {
        float x[2] = {
            t->pos[i].v.x - t->origin.v.x, t->pos[i].v.y - t->origin.v.y
        };
        uint index[2] = {idx(x[0], h32), idx(x[1], h32)};
        if (i == 96089) {
            printf("%f %f %d\n", x[0], t->pos[i].v.x, index[0]);
        }

        // в lists[index.y][index.x] лежит голова списка, отвечающего за область (index.y,index.x)
        // в list[i] сохраняем предыдущий элемент списка, в list[i] пишет только один поток, поэтому синхронизация не нужна
        t->list[i] = t->list_heads[index[1]][index[0]].exchange(i);
    }
}

void parts(Test* t) {
//#pragma omp parallel for num_threads(threads)
#pragma omp parallel for
    for (int i = 0; i < threads; i++) {
        int y = i / nn;
        int x = i % nn;
        parts_thread_init(t, threads, i, x, y);
    }

//#pragma omp parallel for num_threads(threads)
#pragma omp parallel for
    for (int i = 0; i < threads; i++) {
        int y = i / nn;
        int x = i % nn;
        parts_thread(t, threads, i, x, y);
    }
}

void print_lists(Test* t) {
    int maxn = 0;
    int sumn = 0;
    for (int y = 0; y < nn; y++) {
        for (int x = 0; x < nn; x++) {
            printf("%d %d\n", x, y);
            int cur = t->list_heads[y][x];
            int n = 0;
            while (cur != -1) {
                auto& pos = t->pos[cur];
                printf("%d %d > %d %f %f\n", x, y, cur, pos.v.x, pos.v.y);
                cur = t->list[cur];
                n ++;
            }
            printf("count > %d %d %d\n", x, y, n);
            maxn = std::max(maxn, n);
            sumn += n;
        }
    }
    verify(sumn == t->n);
    printf("maxn: %d, sumn: %d\n", maxn, sumn);
}

void parts_sort_thread_init(Test* t, int globalIndex, int x, int y) {
    for (int j = 0; j < nn; j++) {
        t->cells[y][x][j].n = 0;
    }
}

void parts_sort_thread(Test* t, int globalIndex, int x, int y) {
    float l = t->l;
    float h32 = l/nn;

    int cur = t->list_heads[y][x];
    while (cur != -1) {
        auto& pos = t->pos[cur];
        float r[3] = {
            pos.v.x - t->origin.v.x, pos.v.y - t->origin.v.y, pos.v.z - t->origin.v.z
        };
        uint index[3] = {idx(r[0], h32), idx(r[1], h32), idx(r[2], h32)};
        if (index[0] != (uint)x) {
            printf("%d, %d %d, %.16f, %.16f\n", cur, index[0], x, pos.v.x, pos.v.x - t->origin.v.x);
            verify(index[0] == (uint)x);
        }
        if (index[1] != (uint)y) {
            verify(index[1] == (uint)y);
        }
        auto& cell = t->cells[index[2]][index[1]][index[0]];
        verify (cell.n < max_per_cell);
        cell.data[cell.n++] = cur;
        cur = t->list[cur];
    }
}

void parts_sort(Test* t) {
//#pragma omp parallel for num_threads(threads)
#pragma omp parallel for
    for (int i = 0; i < threads; i++) {
        int y = i / nn;
        int x = i % nn;
        parts_sort_thread_init(t, i, x, y);
    }

//#pragma omp parallel for num_threads(threads)
#pragma omp parallel for
    for (int i = 0; i < threads; i++) {
        int y = i / nn;
        int x = i % nn;
        parts_sort_thread(t, i, x, y);
    }
}

void print_cells(Test* t) {
    int maxn = 0;
    int sumn = 0;

    for (int z = 0; z < nn; z++) {
        for (int y = 0; y < nn; y++) {
            for (int x = 0; x < nn; x++) {
                int n = t->cells[z][y][x].n;
                maxn = std::max(maxn, n);
                sumn += n;
            }
        }
    }
    printf("> maxn: %d, sumn: %d\n", maxn, sumn);
    fflush(stdout);

    verify(sumn == t->n);
}

float maxR = 0;
float maxR1 = 0;

void process(Test*t, Index id, Index other, const Pos& dh, int from, int to) {
    int i0,j0;
    Pos ri, rj;

    auto& cell1 = t->cells[id.z][id.y][id.x];
    auto& cell2 = t->cells[other.z][other.y][other.x];
    int n1 = cell1.n;
    int n2 = cell2.n;

    float l = t->l;
    float rcrit = l / 32; // nn;
    float h = l/nn;
    float G = 1;
    const float eps = 0.01; // TODO
    int i = -1;

    for (i0 = from; i0 < to; i0++) {
        i = -1;
        if (i0 < n1) {
            i = cell1.data[i0];
            ri = t->pos[i];
            ri.v.x += dh.v.x;
            ri.v.y += dh.v.y;
            ri.v.z += dh.v.z;
        }

        for (j0 = 0; j0 < n2; j0++) {
            if (i0 < n1 && (id != other || i0 != j0)) {
                rj = t->pos[cell2.data[j0]];
                float mass = rj.v.w;
                float R = 0, Re;
                for (int k = 0; k < 3; k++) {
                    R += (rj.r[k] - ri.r[k]) * (rj.r[k] - ri.r[k]);
                }
                if (R > 3*(2*h)*(2*h)) {
                    printf("%f %f, {%d,%d,%d}, {%d,%d,%d}, %d %d, %d %d, %d %d, %d %d, {%f,%f,%f}, {%f,%f,%f}\n",
                        sqrt(R),
                        sqrt(3*(2*h)*(2*h)),
                        id.x, id.x, id.z,
                        other.x, other.x, other.z,
                        n1, n2,
                        from, to,
                        i0, j0, 
                        cell1.data[i0], cell2.data[j0],
                        t->pos[cell1.data[i0]].r[0], t->pos[cell1.data[i0]].r[1], t->pos[cell1.data[i0]].r[2],
                        t->pos[cell2.data[j0]].r[0], t->pos[cell2.data[j0]].r[1], t->pos[cell2.data[j0]].r[2]
                        );
                    verify(R < 3*(2*h)*(2*h));
                }
                R = sqrt(R);
                Re = R + eps;
                maxR = std::max(maxR, R);
                if (R < rcrit) {
                    maxR1 = std::max(maxR1, R);
                    for (int k = 0; k < 3; k++) {
                        t->F[i].r[k] += - mass * G * (ri.r[k]-rj.r[k])/Re/Re/Re;
                    }
                }
            }
        }
    }
}

void parts_pp_thread(Test* t, int threads, int i, int x, int y, int z) {
    auto& cell = t->cells[z][y][x];
    int n1 = cell.n;
    int work_size = int((n1+threads-1)/threads);
    int from = int(i*work_size);
    int to = from+work_size;
    int i0,k0,j0;

    for (i0 = from; i0 < to && i0 < n1; i0++) {
        Pos v; v.r[0] = v.r[1] = v.r[2] = v.r[3] = 0;
        t->F[cell.data[i0]] = v; // zero
        t->Fl[cell.data[i0]].nf ++;
        verify(t->Fl[cell.data[i0]].nf == 1);
    }

    auto id = Index{x,y,z};
    float l = t->l;

    for (i0 = -1; i0 <= 1; i0++) {
        for (k0 = -1; k0 <= 1; k0++) {
            for (j0 = -1; j0 <= 1; j0++) {
                Pos dh; dh.r[0] = dh.r[1] = dh.r[2] = 0;
                if (z+i0 < 0)   dh.v.z =  l;
                if (z+i0 >= nn) dh.v.z = -l;
                if (y+k0 < 0)   dh.v.y =  l;
                if (y+k0 >= nn) dh.v.y = -l;
                if (x+j0 < 0)   dh.v.x =  l;
                if (x+j0 >= nn) dh.v.x = -l;
                process(t, id, Index{(nn+x+j0)%nn, (nn+y+k0)%nn, (nn+z+i0)%nn}, dh, from, to);
            }
        }
    }
}

void parts_pp(Test* t) {
    // debug
    for (auto& fl : t->Fl) {
        fl.nf = 0;
    }


    for (int z = 0; z < nn; z++) {
        for (int y = 0; y < nn; y++) {
            for (int x = 0; x < nn; x++) {
//#pragma omp parallel for num_threads(nn)
#pragma omp parallel for
                for (int i = 0; i < nn; i++) {
                    parts_pp_thread(t, nn, i, x, y, z);
                }
            }
        }
    }

    float h = t->l / nn;
    printf("maxR: %f, maxR1: %f, theoretical: %f, h: %f\n", maxR, maxR1, sqrt(3*(2*h)*(2*h)), h);

    // debug
    for (auto& fl : t->Fl) {
        verify(fl.nf == 1);
    }
}

void move(Test* t) {
#pragma omp parallel for
    for (int i = 0; i < (int)t->vel.size(); i++) {
        for (int k = 0; k < 3; k++) {
            t->vel[i].r[k] += t->dt * t->F[i].r[k];
            t->pos[i].r[k] += t->dt * t->vel[i].r[k];
            if (t->pos[i].r[k] < t->origin.r[k]) {
                t->pos[i].r[k] += t->l;
            }
            if (t->pos[i].r[k] >= t->origin.r[k] + t->l) {
                t->pos[i].r[k] -= t->l;
            }
        }
    }
}

void plot(Test* t, int step) {
    using namespace fdm;

    printf("Plot %d\n", step);

    char buf[1024];
    snprintf(buf, sizeof(buf), "pp_%07d.png", step);

    matrix_plotter plotter(
        matrix_plotter::settings()
            .sub(1, 1)
            .devname("pngcairo")
            .fname(buf));

    plenv(t->origin.r[1], t->origin.r[1]+t->l, t->origin.r[0], t->origin.r[0]+t->l, 1, 0);
    pllab("Y", "X", "");
    double r = 0.05;
    for (auto& p: t->pos) {
        plarc(p.r[1], p.r[0], r, r, 0, 360, 0, 1);
    }
}

void calc(Test* t) {
    float tt = 0;
    float T = 100;
    int step = 0;
    while (tt < T) {
        printf("Step: %d %f\n", step, tt);
        parts(t);
        parts_sort(t);
        parts_pp(t);
        move(t);
        if (step % 10 == 0) {
            plot(t, step);
        }
        tt += t->dt;
        step ++;
    }
}

int main() {
    Test* t = (Test*)calloc(1, sizeof(*t));
    t->n = 200000;
    t->l = 100;
    t->dt = 0.01;
    t->origin.v.x = t->origin.v.y = t->origin.v.z = -50;
    gen(t);
    t->list.resize(t->n);
    //parts(t);
    //print_lists(t);
    //parts_sort(t);
    //print_cells(t);
    //parts_pp(t);
    calc(t);
    free(t);
    return 0;
}
