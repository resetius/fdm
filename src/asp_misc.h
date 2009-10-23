#ifndef MISC_H
#define MISC_H
/*  $Id$   */

/* Copyright (c) 2004, 2005, 2006 Alexey Ozeritsky
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
/*
 *       Misc Functions
 */

#include <stdio.h>

namespace asp {

	void gramm_matrix(double * g, const double *e1, const double *e2, int m, int n);

	/**
	 * загрузка матрицы из файла формата
	 * @param n размерность
	 * n * n чисел формата double
	 */
	void load_dim_matrix_from_txtfile(double **A, int *n, const char *filename);

	/*! загрузка матрицы из файла формата
	    n * m чисел формата double
		\param n - число строк
		\param m - число столбцов
		A[i * n + j], i = 0,n-1, j = 0,m-1
	*/
	void load_matrix_from_txtfile(double **A, int *n, int *m, const char *filename);
	
	/*! загрузка матрицы из бинарного файла 
	    \param separator - символ разделитель строк
		\param endian    - 1 - little endian, 2 - big endian
		\param n - число строк
		\param m - число столбцов
	*/
	void load_matrix_from_binfile(double **A, int *n, int *m, 
		int endian,
		const char * separator,		
		const char *filename);

	/*! сохранение матрицы в бинарный файл
	    \param separator - символ разделитель строк
		\param endian    - 1 - little endian, 2 - big endian
		\param n - число строк
		\param m - число столбцов
	*/
	void save_matrix_to_binfile(const double *A, int n, int m, 
		int endian,
		const char * separator,		
		const char *filename);

	/*! загрузка строки произвольной длины из файла*/
	char * fget_long_string(FILE *f);
	char * _fget_long_string(FILE *f, const char * separator, int *len);

	/**
	 * загрузка матрицы из функции
	 */
	void load_matrix_from_func(double **A, int n, double (*f)(int i, int j));	
	/**
	 * загрузка из функции или из файла
	 * приоритет по файлу
	 * если оба указателя ноль, то получаем сообщение об ошибке
	 */
	void load_dim_matrix_from_txtfile_or_func(double **A, int *n, const char *filename, double (*f)(int i, int j));
	
	double dist(const double *y1, const double *y2, int n); /*!<евклидова длина*/
	double dist1(const double *y1, const double *y2, int n); /*!<длина в норме 1*/

#define dist2 dist

	double norm1(const double *x,int n); /*!<норма 1 (максимум)*/
	double norm2(const double *x,int n); /*!<евклидова норма*/

	void normalize1(double *x, int n); /*считает x/||x|| по норме1*/
	void normalize2(double *x, int n); /*считает x/||x|| по норме2*/

	/**
	 * Нормализует массив, чтобы данные находились в отрезке [a, b]
	 */
	void normilize_(double *x, double a, double b, int n);

	void noise_data(double *x, double percent, int n);

	/**
	 * Отображение точки отрезка [a1, b1] в точку отрезка [a2, b2]
	 */
	double normalize_point(double x, double a1, double b1, double a2, double b2);
	void normalize_point_(double *x, double a1, double b1, double a2, double b2);

	/**
	 * находит минимум в массиве
	 */
	double find_min(const double *x, int n);

	/**
	 * находит максимум в массиве
	 */
	double find_max(const double *x, int n);

	/**
	 * Матричная экспонента
	 * @param Dest - результат экспонента от A.
	 * @param A - матрица
	 * @param n - размерность
	 */
	void matrix_exp(double * Dest, const double * A, int n);

	/**
	 * Умножение матриц. Умножение безопасное. 
	 * Можно использовать одни и те же параметры для ответа и данных
	 * @param Dest - результат
	 * @param A
	 * @param B
	 * @param n
	 * @return A*B
	 */
	void matrix_mult(double * Dest, const double * A, const double * B, int n);

	/**
	 * Умножение матрицы на вектор . Умножение безопасное. 
	 * Можно использовать одни и те же параметры для ответа и данных
	 * @param Dest - результат
	 * @param A
	 * @param v
	 * @param n
	 * @return A * v
	 */
	void matrix_mult_vector(double * Dest, const double * A, const double * v, int n);
	void matrix_mult_scalar(double * Dest, const double * A, double v, int n);
	void vector_mult_scalar(double * Dest, const double * A, double v, int n);

	/**
	 * Сумма матриц. 
	 * @param Dest - результат
	 * @param A
	 * @param B
	 * @param n
	 * @return A + B
	 */
	void matrix_sum(double * Dest, const double * A, const double * B, int n);
	void vector_sum(double * Dest, const double * A, const double * B, int n);
	void vector_sum1(double * Dest, const double * A, const double * B, double k1, double k2, int n);
	void vector_sum2(double * Dest, const double * A, const double * B, double c, int n);

	/**
	 * Разность векторов. 
	 * @param Dest - результат
	 * @param A
	 * @param B
	 * @param n
	 * @return A - B
	 */
	void vector_diff(double * Dest, const double * A, const double * B, int n);

	/**
	 * скалярное произведение в R^k
	 */
	double scalar(const double *X1, const double *X2, int k);
	/**
	 * скалярное произведение столбцов i, j матриц
	 * @param X - nx * n, Y - ny * n
	 * @param nx, ny - число строк в матрицах
	 */
	double matrix_cols_scalar(const double *X, const double *Y, int i, int j, int nx, int ny, int n);

	/*!линейное отображение вектора \param h
	в вектор \param h1, вектор разлагается по 
	ортогональному базису e,
	размерность пространства n, размерность отображения nx,
	вектора в базисе лежат по столбцам,
	\param L - матрица отображения
	*/
	void linear_reflection(double *h1, const double *h, const double *e, 
		const double *L, int n, int nx);

	void make_identity_matrix(double *X, int n);
	void load_identity_matrix(double **X, int n);
	/* возвращает три средние диагонали матрицы */
	void extract_tdiags_from_matrix(double *D, double *M, double *U, const double*A, int n);
	
	/**
	 * ввод/вывод
	 * параметры
	 * @param f      - файл в который печатаем
	 * @param A      - матрица/вектор, который печатаем
	 * @param n      - размерность матрицы/вектора
	 * @param k      - длинна строки для прямоугольных
	 * @param m      - подстрока/подстолбец, который печатаем 
	 *           (если m = max(n, k) то печатаем полностью)
	 * @param format - форматная строка, которая передается printf'у 
	 */
	/*неформатированный вывод*/
	void printmatrix(const double*A, int n, int m);
	void printvector(const double*A, int n, int m);
	void fprintmatrix(FILE*f, const double*A, int n, int m);
	void fprintvector(FILE*f, const double*A, int n, int m);
	/*форматированный вывод*/
	void printfmatrix(const double*A, int n, int m, const char* format);
	void printfvector(const double*A, int n, int m, const char* format);
	void fprintfmatrix(FILE*f, const double*A, int n, int m, const char* format);
	void _fprintfmatrix(const char *fname, const double*A, int n, int m, const char* format);
	void fprintfvector(FILE*f, const double*A, int n, int m, const char* format);
	void _fprintfvector(const char *fname, const double*A, int n, int m, const char* format);
	/*форматированный с отступами для положительных*/
	void printfvmatrix(const double*A, int n, int m, const char* format);
	void printfvvector(const double*A, int n, int m, const char* format);
	void fprintfvmatrix(FILE*f, const double*, int, int, const char* format);
	void fprintfvvector(FILE*f, const double*, int, int, const char* format);
	/*для прямоугольных матриц*/
	void printfwmatrix         (const double*A, int n, int k, int m, const char* format);
	void fprintfwmatrix(FILE*f, const double*A, int n, int k, int m, const char* format);
	void _fprintfwmatrix(const char *fname, const double*A, int n, int k, int m, const char* format);
	/*целочисленный ввод/вывод*/
	/*неформатированный вывод*/
	void printimatrix(const int *A, int n, int m);
	void printivector(const int *A, int n, int m);
	void fprintimatrix(FILE*f, const int*A, int n, int m);
	void fprintivector(FILE*f, const int*A, int n, int m);
		
	/*целочисленный минимум и максимум двух чисел*/
	int mini(int,int);
	int maxi(int,int);
	
	/*вещественные минимум и максимум двух чисел*/	
	double maxd(double,double);
	double mind(double,double);

	double max3d(double x, double y, double z);
	double min3d(double x, double y, double z);

	int max3i(int x, int y, int z);
	int min3i(int x, int y, int z);
	
	/*целочисленная степень*/
	double ipow(double x, int n);
	int _ipow(int x, int n);
	
	void matrix2file(const double**M, int N1, int N2, const char* fname);
	void _matrix2file(const double *M, int N1, int N2, const char *fname);
	double matrix_diff(const double** M1, const double** M2, int N1, int N2);
	double _matrix_diff(const double *A, const double *B, int N1, int N2);
	double _matrix_inside_diff(const double *A, const double *B, int N1, int N2);

	/*!транспонирование*/
	void matrix_transpose(double *A, int n);
	/*!транспонирование с копированием*/
	void matrix_copy_transpose(double *Dest, const double * Source, int n);
	/* транспонирование с копированием для прямоугольных матриц.
	 * @param Dest   - ответ размерности m * n
	 * @param Source - исходная матрица размерности n * m
	 */
	void matrix_copy_transposew(double * Dest, const double * Source , int n, int m);
	/* транспонирование прямоугольных матриц.
	 * @param A   - матрица размерности m * n
	 */
	void matrix_transposew(double * A, int n, int m);

	/*!заполняет массив случайными числами*/
	void random_initialize(double *a, int n);
	void printcopyright();

	/*!единица на i'м месте*/
	void basis(double *vec, int n, int i);
} /*namespace asp*/
#endif

