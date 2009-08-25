#ifndef _ASP_CHECK_H
#define _ASP_CHECK_H
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
/**
 * Функции проверки
 */

#ifdef __cplusplus
extern "C" {
	namespace asp {
#endif

/**
 * вычисляет нормы ||Ax-lx||
 * @param A  - матрица
 * @param vr - левые собственные вектора
 * @param wr - вещественные части собственных значений
 * @param n  - размерность
 */
	void check_evalues_evectors_real(double *A, double *v, double *wr, int n);
/*!
 * проверка векторов на ортогональность
 * \param v - собственные вектора
 * \param n - размерность пространства
 * \param m - число векторов
 * вектора в матрице v расположены по столбцам:
 * v[i * m + j], i=0,...,n, j=0,...,m
 */
	void check_vectors_ortogonalization(const double *v, int n, int m);
	void check_rows_vectors_ortogonalization(const double *v, int n, int m);

/**
 * проверка подпространств на ортогональность
 * @param Ex - собственные вектора
 * @param Ey - собственные вектора
 * @param nx - размерность Ex
 * @param ny - размерность Ey
 * @param n  - размерность всего пространства
 */
	int check_subspaces_ortogonalization(double *Ex, double *Ey, int nx, int ny, int n);	

/**
 * проверка векторов на линейную независимость
 * n векторов, размерности n
 * возвращает ранк
 * @param v - вектора
 * @param n - размерность
 */
	int check_linear_independence(double *v, int n);

	/*проверка алгоритма Грамма-Шмидта*/
	void check_gsch_(int n, int m, void (*gsch)(double *, int, int));
	void check_gsch(int n, int m);
	void check_mgsch(int n, int m);

	/*проверка ортогонализации по строкам*/
	void check_rows_gsch_(int n, int m, void (*gsch)(double *, int, int));
	void check_rows_gsch(int n, int m);
	void check_rows_mgsch(int n, int m);

#ifdef __cplusplus
}
} /*namespace asp*/
#endif
#endif //#ifndef _ASP_CHECK_H
