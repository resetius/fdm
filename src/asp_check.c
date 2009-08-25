/* $Id$ */

/* Copyright (c) 2005, 2006 Alexey Ozeritsky
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by Alexey Ozeritsky.
 * 4. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission
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
/** Функции проверки. 
 * Функции проверки алгоритмов из asplib
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "asp_lib.h"

/**
 * вычисляет нормы \f$\|Ax-lx\|\f$
 * @param A  - матрица
 * @param vr - левые собственные вектора
 * @param wr - вещественные части собственных значений
 * @param n  - размерность
 */
void check_evalues_evectors_real(double *A, double *v, double *wr, int n) {
	int i,j,m;
	double sum;
	double norm2m;
	printf("check_evalues_evectors_real...\n");
	/**
	 * вычисляемм нормы \f$\|Ax_m-l_m x_m\| = \|a-b\|\f$
	 * \f$x_mj = v[j*n+m]\f$ хранится по столбцам
	 * \f$y    = x_m\f$
	 * \f$a_i  = \sum A_{ij} y_j  b_i = l_m y_i\f$
	 * для всех \f$i\f$ считаем 
	 * sum \f$A_{ij} y_j - l_m y_i\f$
	 */
	for (m = 0; m < n; m++) {
		norm2m = 0.0; //норма разности
		for (i = 0; i < n; i++) {
			sum = 0.0; //a_i			
			for (j = 0; j < n ; j++) {
				sum += A[i*n+j]*v[j*n+m];
			}
			sum -= wr[m]*v[i*n+m];
			norm2m += sum*sum;
		}
		norm2m = sqrt(norm2m);
		printf("%d %lf\n",m,norm2m);
	}
}

/*!
 * проверка векторов на ортогональность
 * \param v - собственные вектора
 * \param n - размерность пространства
 * \param m - число векторов
 * вектора в матрице v расположены по столбцам:
 * v[i * m + j], i=0,...,n, j=0,...,m
 */
void check_vectors_ortogonalization(const double *v, int n, int m) {
	int i, j, k;
	int l = 10; /*ограничение*/
	/**
	 * вычисляем скалярные произведения
	 * (v_i, v_j)
	 * v_i = v[k * m + i]
	 */
	/*по векторам*/
	for (i = 0; i < m && i < l; ++i) {
		/*по векторам*/
		for (j = 0; j < m && j < l; ++j) {
			double sum = 0.0;
			/*по координатам*/
			for (k = 0; k < n; ++k) {
				sum += v[k * m + i] * v[k * m + j];
			}
			//printf("(v_%d, v_%d)=%.1lf ", i, j, sum);
			printf("%4.1lf ", sum);
		}
		printf("\n");
	}
	printf("\n");
}

/*тоже самое вектора хранятся по строкам*/
void check_rows_vectors_ortogonalization(const double *v, int n, int m) {
	int i, j, k;
	int l = 10; /*ограничение*/
	/**
	 * вычисляем скалярные произведения
	 * (v_i, v_j)
	 * v_i = v[k * m + i]
	 */
	/*по векторам*/
	for (i = 0; i < m && i < l; ++i) {
		/*по векторам*/
		for (j = 0; j < m && j < l; ++j) {
			double sum = 0.0;
			/*по координатам*/
			for (k = 0; k < n; ++k) {
				sum += v[i * n + k] * v[j * n + k];
			}
			//printf("(v_%d, v_%d)=%.1lf ", i, j, sum);
			printf("%4.1lf ", sum);
		}
		printf("\n");
	}
	printf("\n");
}

/*проверка алгоритма Грамма-Шмидта*/
void check_gsch_(int n, int m, void (*gsch)(double *, int, int))
{
	double * v = malloc(n * m * sizeof(double));
	random_initialize(v, n * m);

	printf("before gsch:\n");
	check_vectors_ortogonalization(v, n, m);
	gsch(v, n, m);
	printf("after gsch:\n");
	check_vectors_ortogonalization(v, n, m);
	free(v);
}

void check_rows_gsch_(int n, int m, void (*gsch)(double *, int, int))
{
	double * v = malloc(n * m * sizeof(double));
	random_initialize(v, n * m);

	printf("before gsch:\n");
	check_rows_vectors_ortogonalization(v, n, m);
	gsch(v, n, m);
	printf("after gsch:\n");
	check_rows_vectors_ortogonalization(v, n, m);
	free(v);
}

void check_gsch(int n, int m)
{
	printf("checking gsch (%d, %d):\n", n, m);
	check_gsch_(n, m, ortogonalize_cols_gsch);
	printf("done\n");
}

void check_mgsch(int n, int m)
{
	printf("checking gsch (%d, %d):\n", n, m);
	check_gsch_(n, m, ortogonalize_cols_mgsch);
	printf("done\n");
}

void check_rows_gsch(int n, int m)
{
	printf("checking gsch (%d, %d):\n", n, m);
	check_rows_gsch_(n, m, ortogonalize_gsch);
	printf("done\n");
}

void check_rows_mgsch(int n, int m)
{
	printf("checking gsch (%d, %d):\n", n, m);
	check_rows_gsch_(n, m, ortogonalize_mgsch);
	printf("done\n");
}

/**
 * проверка подпространств на ортогональность
 * @param Ex - собственные вектора
 * @param Ey - собственные вектора
 * @param nx - размерность Ex
 * @param ny - размерность Ey
 * @param n  - размерность всего пространства
 */
int check_subspaces_ortogonalization(double *Ex, double *Ey, int nx, int ny, int n) {
	int i, j;
	int error = 0;
	double scal;
	/**
	 * надо проверить все пары (u_i, v_j) для u_i из Ex, v_j из Ey
	 */
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			scal = matrix_cols_scalar(Ex, Ey, i, j, nx, ny, n);
			if (fabs(scal) > EPS32) {
				error += 1;
			}
		}
	}
	return error;
}

/**
 * проверка векторов на линейную независимость
 * n векторов, размерности n
 * возврашает ранк
 * @param A - вектора
 * @param n - размерность
 */
int check_linear_independence(double *A, int n) {
	int i, j, k, r;
    for(j = 0; j < n; j++) {
    	double a;
     /*поиск главного значения*/
        if (findmain(&r, j, n, A) < EPS32) { return j; }
        swop(r, j, n, A);

        a = A[j * n + j];
        for(i = j + 1; i < n; i++) {
            double ba = A[i * n + j] / a;

            for(k = j; k < n; k++)
                A[i * n + k] -= A[j * n + k] * ba;
        }
    }
    return j;
}

