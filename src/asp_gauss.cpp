/* Copyright (c) 2000, 2001, 2002, 2003, 2004, 2022 Alexey Ozeritsky
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** Алгориьтмы линейной алгербры.
 * Решение линейных систем и обращения матриц, мои алгоритмы
 * и обвязка для lapack
 */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "asp_macros.h"
#include "asp_gauss.h"
#include "asp_misc.h"

using namespace asp;

static void gauss_reverse (const double *A, const double *b, double *x, int n, int diagUnit)
{
	int j, k;

	for (k = n - 1; k >= 0; k--) {
		x[k] = b[k];
		for (j = k + 1; j < n; j++) {
			x[k] = x[k] - x[j] * A[k*n+j];
		}
		if (!diagUnit) {
			x[k] = x[k] / A[k*n+k];
		}
	}
}

int gauss (const double *A_, const double *b_, double *x, int n)
{
	int i, j, k;
	double p;
	int imax;
	double Eps = 1.e-15;

	double *A = (double*)malloc (n * n * sizeof(double));
	memcpy(A, A_, n * n * sizeof(double));

	double *b = (double*)malloc(n * sizeof(double));
	memcpy(b, b_, n * sizeof(double));

	for (k = 0; k < n; k++) {
		imax = k;

		for (i = k + 1; i < n; i++) {
			if (fabs(A[i*n+k]) > fabs(A[imax*n+k])) imax = i;
		}

		for (j = k; j < n; j++) {
			p = A[imax*n+j];
			A[imax*n+j] = A[k*n+j];
			A[k*n+j] = p;
		}
		p = b[imax];
		b[imax] = b[k];
		b[k] = p;

		p = A[k*n+k];

		if (fabs(p) < Eps) {
			printf("Warning in %s %s : Near-null zero element\n", __FILE__, __FUNCTION__);
			return -1;
		}

		for (j = k; j < n; j++) {
			A[k*n+j] = A[k*n+j] / p;
		}
		b[k] = b[k] / p;

		for (i = k + 1; i < n; i++) {
			p = A[i*n+k];
			for (j = k; j < n; j++) {
				A[i*n+j] = A[i*n+j] - A[k*n+j] * p;
			}
			b[i] = b[i] - b[k] * p;
		}
	}

	gauss_reverse(A, b, x, n, true);

	free(b); free(A);

	return 0;
}

/**
 * Обращение матрицы.
 * @param n размерность
 * @param A исходная
 * @param X обратная
 * @return
 */
void inverse_general_matrix_my(double *Dest, double * Source, int n) {
    int i, j, k;
    int r;
	double * A = (double*) malloc(n * n * sizeof(double));
	double * X;
	int need_to_free_X = 0;

	NOMEM(A);
	memcpy(A, Source, n * n * sizeof(double));


	if (Dest == Source) {
		load_identity_matrix(&X, n);
		need_to_free_X = 1;
	} else {
		X = Dest;
		make_identity_matrix(X, n);
	}

    for (j = 0; j < n; j++) {
    	double a;
     /*поиск главного значения*/
        findmain(&r,j,n,A);
        swop(r,j,n,A);
        swop2(r,j,n,X);

        a = A[j * n + j];
        for (i = j + 1; i < n; i++) {
            double ba = A[i * n + j] / a;

            for (k = j; k < n; k++)
                A[i * n + k] -= A[j * n + k] * ba;

            for (k = 0; k < n; k++)
                X[i * n + k] -= X[j * n + k] * ba;
        }
    }

    for (j = n - 1; j >= 0; j--) {
        double a = A[j * n + j];
        for (i = j - 1; i >= 0; i--) {
            double ba = A[i * n + j] / a;
            for (k = n - 1; k >= j; k--)
                A[i * n + k] -= A[j * n + k] * ba;

            for (k = n - 1; k >= 0; k--)
                X[i * n + k] -= X[j * n + k] * ba;
        }
    }

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            X[i * n + j] *= 1.0 / A[i * n + i];
        }
    }

	if (need_to_free_X) {
		memcpy(Dest, X, n * n * sizeof(double));
		free(X);
	}
	free(A);
}

void  solve_tdiag_linear_my(double * B, double*A1, double *A2, double *A3, int n) {
	int i, j = 0;
	double a;

	for (j = 0; j < n - 1; j++) {
		a = A1[j] / A2[j];
		A1[j] = 0;
		A2[j + 1] -= A3[j] * a;
		B[j + 1] -= B[j] * a;
	}

	for (j = n - 1; j > 0; j--) {
//		if (fabs(A2[j]) > 1e-16) {
			a = A3[j - 1] / A2[j];
			B[j - 1] -= B[j] * a;
//		}
	}

	for (i = 0; i < n; i++) {
//		if (fabs(A2[i]) > 1e-16)
			B[i] *= 1 / (A2[i]);
	}
}

void  solve_tdiag_linearf_my(float * B, float*A1, float *A2, float *A3, int n) {
	int i, j = 0;
	float a;

	for (j = 0; j < n - 1; j++) {
		a = A1[j] / A2[j];
		A1[j] = 0;
		A2[j + 1] -= A3[j] * a;
		B[j + 1] -= B[j] * a;
	}

	for (j = n - 1; j > 0; j--) {
		a = A3[j - 1] / A2[j];
		B[j - 1] -= B[j] * a;
	}

	for (i = 0; i < n; i++) {
		B[i] *= 1 / (A2[i]);
	}
}

void inverse_tdiag_matrix_my(double *Dest, double * A1,double *A2,double *A3, int n) {
	int i, j = 0, k;
	double a;
	double *X = Dest;

	make_identity_matrix(X, n);

	for (j = 0; j < n - 1; j++) {
		a = A1[j] / A2[j];
		A1[j] = 0;
		A2[j + 1] -= A3[j] * a;
		//for (i = j + 1; i < n; i++) {
		//	for (k = 0; k < n; k++)
		//		X[i * n + k] -= X[j * n + k] * a;
		//}
		for (k = 0; k < n; k++)
			X[(j + 1) * n + k] -= X[j * n + k] * a;
		//B[j + 1] -= B[j] * a;
	}

	for (j = n - 1; j > 0; j--) {
		a = A3[j - 1] / A2[j];
		//for (i = j - 1; i >= 0; i--) {
		//	for (k = n - 1; k >= 0; k--)
		//		X[i * n + k] -= X[j * n + k] * a;
		//}
		for (k = n - 1; k >= 0; k--)
			X[(j - 1) * n + k] -= X[j * n + k] * a;

		//B[j - 1] -= B[j] * a;
	}

	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			X[i * n + j] *= 1.0 / A2[i];
		}
	}
	//for (i = 0; i < n; i++)
	//	B[i] *= 1 / (A2[i]);
}

void inverse_tdiag_matrix2_my(double *Dest, double * Source, int n) {
	double * X = 0;
	double * D = 0, * M = 0, * U = 0;
	int need_to_free_X = 0;

	if (Dest == Source) {
		load_identity_matrix(&X, n);
		need_to_free_X = 1;
	} else {
		X = Dest;
		make_identity_matrix(X, n);
	}

	D    = (double *) malloc(n * sizeof(double)); NOMEM(D);
	M    = (double *) malloc(n * sizeof(double)); NOMEM(M);
	U    = (double *) malloc(n * sizeof(double)); NOMEM(U);

	//make_identity_matrix(X, n);
	extract_tdiags_from_matrix(D, M, U, Source, n);
	inverse_tdiag_matrix_my(X, D, M, U, n);
	free(D); free(M); free(U);
	if (need_to_free_X) {
		memcpy(Dest, X, n * n * sizeof(double));
		free(X);
	}
}

/**
 * поиск главного значения по столбцу
 */
double findmain(int*r, int k, int n, double*A) {
	int j;
	double max = A[k * n + k];
	double c;
	(*r)=k;
	for (j = k; j < n; j++) {
		c = fabs(A[j*n+k]);
		if (max < c) {
			(*r) = j;
			max  = c;
		}
	}
	return max;
}

/**
 * перестановка k'й и r'й строки
 * начиная с индекса k
 */
void  swop(int r,int k,int n,double*A) {
	int i;
	double temp;
	for(i = k; i < n; i++) {
		temp = A[k*n+i]; A[k*n+i] = A[r*n+i]; A[r*n+i] = temp;
	}
}

/**
 * перестановка k'й и r'й строки
 * начиная с индекса 0
 */
void  swop2(int r,int k,int n,double*A) {
	int i;
	double temp;
	if(r!=k)
		for(i=0;i<n;i++) {
			temp=A[k*n+i];A[k*n+i]=A[r*n+i];A[r*n+i]=temp;
		}
}

double nev(double*A,double*B,int n) {
    int i,j,k;
    //double*C;
    double max=0,temp,cij;
    for(i=0;i<n;i++) {
        temp=0;
        for(j=0;j<n;j++) {
            cij=0;

            for(k=0;k<n;k++) cij+=A[i*n+k]*B[k*n+j];
            if(i==j) cij-=1;
            temp+=fabs(cij);
        }
        if(max<temp) max=temp;
    }
    return max;
}
