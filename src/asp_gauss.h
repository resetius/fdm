#ifndef _GAUSS_H
#define _GAUSS_H
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

/**
 *  Решение линейных систем
 */

/**
 * ни одна из функций не должна изменять исходных значений!
 * смотри требования в sds.h
 * функции с суффиксом _my  мои
 * с суффиксами _sla, _ela, _la используют lapack
 * _sla - simple driver
 * _ela - expert driver
 */

/**
 * Настройки -- какие функции использовать
 * функции оканчивающиеся на _my это моя реализация,
 *         оканчивающиеся на _sla это из библиотеки LAPACK,
 */
#define solve_tdiag_linear(A, B, C, D, n)   solve_tdiag_linear_my(A, B, C, D, n);

#ifdef __cplusplus
extern "C" {
#endif
	int gauss (const double *A_, const double *b_, double *x, int n);

	/* решение трехдиаг линейной системы */
	/* методом гаусса */
	void solve_tdiag_linear_my(double *Dest, double * Down,double *Middle,double *Up, int n);
	void solve_tdiag_linearf_my(float *Dest, float * Down,float *Middle,float *Up, int n);
	void inverse_tdiag_matrix_my(double *Dest, double * Down,double *Middle,double *Up, int n);
	/* обращение матрицы общего вида */
	/* методом гаусса */
	void inverse_general_matrix_my(double * Dest,double * Source, int n);

	void inverse_tdiag_matrix2_my(double *Dest, double * Source, int n);

/*int max1(int,int);*/
	double*matvect(double*,double*,int);
	double*matmat(double*,double*,int);
	double findmain(int*,int,int,double*);
	void swop(int,int,int,double*);
	void swop2(int,int,int,double*);
	double mod(double);
	double nev(double*,double*,int);
/* lapack part */
	void dgesv_(int *n, int *nrhs, double *A, int *lda, int *ipiv, double *X, int *ldb, int *info);
	void dposv_(char *uplo, int *n, int *nrhs, double *A, int *lda, double *X, int *ldb, int *info);
	void dgtsv_(int *n, int *nrhs, double *Down, double *Middle, double *Up, double *X,
		int *ldb, int *info);
	void dptsv_(int *n, int *nrhs, double *D, double *Sub_D, double *X, int *ldb, int *info);
#ifdef __cplusplus
}
#endif
#endif /*_GAUSS_H*/
