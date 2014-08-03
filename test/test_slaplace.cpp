#define _USE_MATH_DEFINES
#include <vector>
#include <string>
#include <algorithm>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "asp_misc.h"
#include "asp_sphere_lapl.h"

using namespace std;

void do_all(const char * fname)
{
	double * f;
	string str1 = fname; str1 += "_lapl.txt";
	string str2 = fname; str2 += "_lapl_1.txt";
	int n_la;
	int n_phi;

	asp::load_matrix_from_txtfile(&f, &n_phi, &n_la, fname);
	SLaplacian l(n_phi, n_la, false);

	vector < double > tmp(n_la * n_phi);
	memset(f, 0, n_la * sizeof(double));

	l.lapl(&tmp[0], f);
	asp::_fprintfwmatrix(str1.c_str(), &tmp[0], n_phi, n_la, std::max(n_phi, n_la), "%.16le ");
	l.lapl_1(&tmp[0], f);
	asp::_fprintfwmatrix(str2.c_str(), &tmp[0], n_phi, n_la, std::max(n_phi, n_la), "%.16le ");

	free(f);
}

int main(int argc, char * argv[])
{
	if (argc > 1) {
		do_all(argv[1]);
	} else {
		fprintf(stderr, "filename required!\n");
	}
	return 0;
}

