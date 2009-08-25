#ifndef _ASP_PROJECTION_H
#define _ASP_PROJECTION_H

/* $Id$ */

/* Copyright (c) 2005 Alexey Ozeritsky
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
 * Алгоритмы проектирования.
 * Проектирование вдоль подпространства, ортогональное проектирование,
 * проектирование как нахождение минимизирующего вектора (lapack'based),
 * Ортогонализация Грамма-Шмидта
 */

#ifdef __cplusplus
extern "C" {
	namespace asp {
#endif

/**
 * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
 * где \f$(e_i, e_j) = \delta_i^j \f$
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - ортонормированный базис подпространства, записанный по столбцам
 * @param n  - размерность пространства
 * @param m  - размерность подпространства
 */
	void ortoproj_ortonormbasis(double *h1, double *h, double *e, int n, int m);

/**
 * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
 * где \f$(e_i, e_j) = 0\f$, при \f$i != j\f$
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - ортонормированный базис подпространства, записанный по столбцам
 * @param n  - размерность пространства
 * @param m  - размерность подпространства
 */
	void ortoproj_ortobasis(double *h1, double *h, double *e, int n, int m);

/**
 * LAPACK'based
 * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
 * минимизируем \f$\|h - e h_1\|\f$
 * используя QR разложение
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - ортонормированный базис подпространства, записанный по столбцам
 * @param n  - размерность пространства
 * @param m  - размерность подпространства
 */
	void ortoproj_lls_qr_la(double *h1, double *h, double *e, int n, int m);

/**
 * LAPACK'based
 * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
 * минимизируем \f$\|h - e h_1\|\f$
 * используя complete orthogonal factorization
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - ортонормированный базис подпространства, записанный по столбцам
 * @param n  - размерность пространства
 * @param m  - размерность подпространства
 */
	void ortoproj_lls_cof_la(double *h1, double *h, double *e, int n, int m);

/**
 * LAPACK'based
 * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
 * минимизируем \f$\|h - e h_1\|\f$
 * используя SVD
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - ортонормированный базис подпространства, записанный по столбцам
 * @param n  - размерность пространства
 * @param m  - размерность подпространства
 */
	void ortoproj_lls_svd_la(double *h1, double *h, double *e, int n, int m);

/**
 * LAPACK'based
 * ортопроекция на подпространство \f$<e_1,...,e_m>\f$
 * минимизируем \f$\|h - e h_1\|\f$
 * используя divide-and-conquer SVD
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - ортонормированный базис подпространства, записанный по столбцам
 * @param n  - размерность пространства
 * @param m  - размерность подпространства
 */
	void ortoproj_lls_dacsvd_la(double *h1, double *h, double *e, int n, int m);

/**
 * ортопроекция на подпространство \f$e_y=<e_1,...,e_{n_y}>\f$ пространства \f$<e_1,...,e_n>\f$
 * @param h  - вектор который проецируем
 * @param h1 - проекция
 * @param e  - базис подпространства на которое проецируем
 * @param ete - матрица обратная к \f$(e_i, e_j)\f$ если 0, то вычисляем
 */
void ortoproj_general(double *h1, double *h, double *e, 
					 double * ete, int m, int n);

	/**
	 * проекция на подпространство ex=<ek,...,el> пространства <e1,...,en>
	 * (l - k) = m;
	 * @param h  - вектор который проецируем
	 * @param h1 - проекция
	 * @param e  - базис ВСЕГО пространства, записанный по столбцам
	 * @param e1 - обратная матрица к e, если 0, то вычисляем
	 * @param n  - размерность пространства
	 * @param m  - размерность подпространства
	 */
	void projection(double *h1, double *h, double *e, 
		double *e1, int n, int k, int l);

	/**
	 * проекция на подпространство ex=<ek,...,el> пространства <e1,...,en>
	 * проекция неортогональная. вектора - собственные вектора матрицы
	 * (l - k) = m;
	 * @param h  - вектор который проецируем
	 * @param h1 - проекция
	 * @param e  - базис подпространства на которое проецируем
	 * @param et - сопряженный базис (с. в. транспонированной матрицы)
	 * @param ete - матрица обратная к \f$(e_i, et_j)\f$ если 0, то вычисляем
	 * @param m   - размерность подпространства проектирования
	 * @param n   - размерность всего пространства
	 */
	void projection2(double *h1, const double *h, const double *e, 
		const double *et, double *ete,
		int m, int n);
	void projection2_ext(double *c, const double *h, const double *e, 
		const double *et,
		double * ete, int m, int n);

	/**
	 * Процесс ортогонализации Грамма-Шмидта
	 * основная формула:
	 * \f$E_{k+1}=e_{k+1}-\sum_{j=1}^{k} \lambda_j E_j\f$,
	 * \f$E_{k+1}=E_{k+1}/||E_{k+1}||\f$ - нормировка,
	 * где \f$\lambda_j=(E_j,e_{k+1})\f$,
	 * \f${E_i}\f$ - новый базис (ортонормированный),
	 * \f${e_i}\f$ - старый базис(неортонормированный)
	 * 
	 * @param X - вектора, которые иртогонализуем, хранящиеся по строкам
	 * @param n - размерность пространства
	 * @param m - число векторов
	 */
	void ortogonalize_gsch(double *X, int n, int m);
	double volume(double *X, double *l, int n, int m);

	/**
	 * Процесс ортогонализации Грамма-Шмидта
	 * с хранением векторов по стобцам
	 * 
	 * @param X - вектора, которые иртогонализуем, хранящиеся по столбцам
	 * @param n - размерность пространства
	 * @param m - число векторов
	 */
	void ortogonalize_cols_gsch(double *X, int n, int m);
	
	/**
	 * Улучшенный алгоритм Грамма-Шмидта.
	 * см. Голуб "Матричные Вычисления" или 
	 * Деммель "Applied Numerical Linear Algebra".
	 * Вектора хранятся по столбцам
	 * @param n - размерность пространства
	 * @param m - число векторов
	 */
	void ortogonalize_cols_mgsch(double *A, int n, int m);
	void ortogonalize_mgsch(double *A, int n, int m);

//lapack
	void dgels_(char * trans, int *m, int *n, int *nrhs, double *a, 
		int *lda, double *b, int *ldb, double *work, int *lwork, int *info);
#ifdef __cplusplus
}
} /*namespace asp*/
#endif
#endif //_ASP_PROJECTION_H
