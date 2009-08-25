/*$Id$*/

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
 * 3. The name of the author may not be used to endorse or promote products
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "asp_interpolate.h"

/*взято и переделано c http://alglib.sources.ru/interpolation/bicubicresample.php*/
/*************************************************************************
Ресэмплирование бикубическим сплайном.

    Процедура   получает   значения   функции    на     сетке
OldWidth*OldHeight и путем интерполяции бикубическим сплайном
вычисляет значения функции в узлах  сетки  размером NewWidth*
NewHeight. Новая  сетка  может  быть как более, так  и  менее
плотная, чем старая.

Входные параметры:
    OldWidth    - старый размер сетки
    OldHeight   - старый размер сетки
    NewWidth    - новый размер сетки
    NewHeight   - новый размер сетки
    A           - массив значений функции на старой сетке.
                  Нумерация элементов [0..OldHeight-1,
                  0..OldWidth-1]

Выходные параметры:
    B           - массив значений функции на новой сетке.
                  Нумерация элементов [0..NewHeight-1,
                  0..NewWidth-1]

Допустимые значения параметров:
    OldWidth>1, OldHeight>1
    NewWidth>1, NewHeight>1
*************************************************************************/
#ifndef max
#define max(a, b) ((a)>(b))?(a):(b)
#endif
void bicubic_resample(int oldwidth,
     int oldheight,
     int newwidth,
     int newheight,
     const double * a,
     double * b)
{
    double * buf;
    double * tbl;
    double * x;
    double * y;
    int mw, mh;
	int m;
    int i, j;

    mw = max(oldwidth, newwidth);
    mh = max(oldheight, newheight);
	m  = max(mw, mh);

	buf = (double*)malloc(mh * mw * sizeof(double));
	x   = (double*)malloc(m * sizeof(double));
	y   = (double*)malloc(m * sizeof(double));
	tbl = (double*)malloc(5 * (max(oldwidth, oldheight) + 1) * sizeof(double));

	for(i = 0; i <= oldheight-1; i++)
    {
        for(j = 0; j <= oldwidth-1; j++)
        {
			//fprintf(stderr, "%d:%d:%d\n", m, i, j);
            x[j] = (double) j/(double) (oldwidth-1);
            y[j] = a[i * oldwidth + j];
        }
		spline3buildtable(oldwidth, x, y, tbl);
        for(j = 0; j <= newwidth-1; j++)
        {
            buf[i * newwidth + j] = spline3interpolate(oldwidth
				, tbl, x, (double) j/(double)(newwidth-1));
		}
    }

    for (j = 0; j <= newwidth-1; j++)
    {
        for (i = 0; i <= oldheight-1; i++)
        {
            x[i] = (double) i /(double)(oldheight-1);
            y[i] = buf[i * newwidth + j];
        }
		spline3buildtable(oldheight, x, y, tbl);
        for (i = 0; i <= newheight-1; i++)
        {
			b[i * newwidth + j] = spline3interpolate(oldheight
				, tbl, x, (double) i/(double)(newheight-1));
        }
    }

	free(buf); free(x); free(y); free(tbl);
}

/*************************************************************************
Билинейное ресэмплирование.

    Процедура   получает   значения   функции    на     сетке
OldWidth*OldHeight и путем билинейной интерполяции  вычисляет
значения функции в узлах  сетки  размером NewWidth*NewHeight.
Новая  сетка  может  быть как более, так и менее плотная, чем
старая.

Входные параметры:
    OldWidth    - старый размер сетки
    OldHeight   - старый размер сетки
    NewWidth    - новый размер сетки
    NewHeight   - новый размер сетки
    A           - массив значений функции на старой сетке.
                  Нумерация элементов [0..OldHeight-1,
                  0..OldWidth-1]

Выходные параметры:
    B           - массив значений функции на новой сетке.
                  Нумерация элементов [0..NewHeight-1,
                  0..NewWidth-1]

Допустимые значения параметров:
    OldWidth>1, OldHeight>1
    NewWidth>1, NewHeight>1
*************************************************************************/
void bilinear_resample(int oldwidth,
     int oldheight,
     int newwidth,
     int newheight,
     const double * a,
     double * b)
{
    int i, j;
    int l;
    int c;
    double t;
    double u;
    double tmp;

    for(i = 0; i <= newheight-1; i++)
    {
        for(j = 0; j <= newwidth-1; j++)
        {
            tmp = (double) i/(double)(newheight-1)*(oldheight-1);
            l = (int)floor(tmp);
            if( l<0 )
            {
                l = 0;
            }
            else
            {
                if( l>=oldheight-1 )
                {
                    l = oldheight-2;
                }
            }
            u = tmp-l;
            tmp = (double) j/(double)(newwidth-1)*(oldwidth-1);
            c = (int)floor(tmp);
            if( c<0 )
            {
                c = 0;
            }
            else
            {
                if( c>=oldwidth-1 )
                {
                    c = oldwidth-2;
                }
            }
            t = tmp-c;
            b[i * newwidth + j] = (1.0-t)*(1.0-u)*
				a[l * oldwidth + c]+t*(1-u)*
				a[l * oldwidth + c+1]+
				t*u*
				a[(l+1) * oldwidth + c+1]+(1-t)*u*
				a[(l+1) * oldwidth + c];
        }
    }
}

void spline3buildtable(int n,
					   double * x,
					   double * y,
					   double * C)
{	
    double *alpha = (double*) malloc(n * sizeof(double));
    double *beta  = (double*) malloc(n * sizeof(double));
	double *d     = (double*) malloc(n * sizeof(double));

    double CAA = 0;
	double fi, ai, bi, ci;	
    int i = 0;

	n--;
    /**
     * Метод прогонки.
     * А. А. Самарский "Теория разностных схем"  - М. Наука 1989, c. 35
     */
    alpha[1] = -0.5;
    beta[1]  = 1.5 * (y[1] - y[0]) / (x[1] - x[0]);

	for (i = 1; i <= n - 1; i++) {
		ai = (x[i + 1] - x[i]);
		bi = (x[i] - x[i - 1]);
		ci = -2.0 * (x[i + 1] - x[i - 1]);
		fi = -(3.0 * (y[i] - y[i - 1]) / (x[i] - x[i - 1]) * (x[i + 1] - x[i])+ 
			3.0 * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (x[i] - x[i - 1]));

        CAA = 1.0 / ci - alpha[i] * ai; /*C_i -alpha_i A_i*/
        alpha[i + 1] = bi * CAA;
        beta[i + 1]  = (ai * beta[i] + fi) * CAA;
    }

    d[n] = (1.5 * (y[n] - y[n - 1]) / (x[n] - x[n - 1])
		- 0.5 * beta[n]) / (1.0 + 0.5 * alpha[n]);

    for (i = n - 1; i >= 0; i--) {
        d[i] = alpha[i + 1] * d[i + 1] + beta[i + 1];
    }

	n++;
    for (i = 0; i < n - 1; i++) {
        C[i]     = y[i];
        C[n+i]   = d[i];
        C[2*n+i] = (3. * (y[i + 1] - y[i]) / (x[i + 1] - x[i])
			- 2. * d[i] - d[i + 1]) / (x[i + 1] - x[i]);
        C[3*n+i] = (d[i] + d[i + 1] 
		- 2. * (y[i + 1] - y[i]) / (x[i + 1] - x[i])) 
			/ ((x[i + 1] - x[i]) * (x[i + 1] - x[i]));
    }

	free(alpha); free(beta); free(d);
}

/*для периодического сплайна*/
void splineper3buildtable(int n,
						  double *y,
						  double *C)
{
    double *alpha = (double*)malloc((n + 2) * sizeof(double));
    double *beta  = (double*)malloc((n + 2) * sizeof(double));
    double *gamma = (double*)malloc((n + 2) * sizeof(double));
    double *u     = (double*)malloc((n + 2) * sizeof(double));
    double *v     = (double*)malloc((n + 2) * sizeof(double));
	double *d     = (double*)malloc((n + 2) * sizeof(double));
    double CAA = 0;
    int i = 0;

    /**
     * Метод циклической прогонки
     * А.А. Самарский. Е.С. Николаев.
     * Методы решений сеточных уравнений. М., Наука, 1978, с. 86
     */
    //-ai y(i-1) + ci yi - bi y(i+1) = fi
    /**
    * производная по x -- dx
    */
    alpha[2] = -0.25; //b1/c1
    beta[2]  = 0.75 * (y[2] - y[0]); //f1/c1
    gamma[2] = -0.25; //a1/c1

    for (i = 2; i <= n; i++) {
        CAA = 4.0 + alpha[i];
        alpha[i + 1] = -1.0 / CAA;
        beta[i + 1]  = (3.0 * (y[(i + 1) % n] - y[i - 1]) - beta[i]) / CAA;
        gamma[i + 1] = -gamma[i] / CAA;
    }

    u[n - 1] = beta[n];
    v[n - 1] = alpha[n] + gamma[n];
    for (i = n - 2; i >= 1; i--) {
        u[i] = alpha[i + 1] * u[i + 1] + beta[i + 1];
        v[i] = alpha[i + 1] * v[i + 1] + gamma[i + 1];
    }

    d[0] = (beta[n + 1] + alpha[n + 1] * u[1]) / 
		(1.0 - gamma[n + 1] - alpha[n + 1] * v[1]);
    for (i = 1; i <= n - 1; i++) {
        d[i] = u[i] + d[0] * v[i];
    }

    for (i = 0; i < n - 1; i++) {
        C[i]     = y[i];
        C[n+i]   = d[i];
        C[2*n+i] = (3. * (y[(i + 1) % n] - y[i])
			- 2. * d[i] - d[i + 1]);
        C[3*n+i] = (d[i] + d[i + 1] - 2. * (y[(i + 1) % n] - y[i]));
    }

    i = n - 1;
    C[i]     = y[i];
    C[n+i]   = d[i];
	C[2*n+i] = (3. * (y[(i + 1) % n] - y[i]) - 2. * d[i] - d[0]);
	C[3*n+i] = (d[i] + d[0] - 2. * (y[(i + 1) % n] - y[i]));
}

double spline3interpolate(int n, const double* c, const double *x, 
						  double t)
{
    double dd, dd2, dd3;
	int k, i = -1;
	for (k = 0; k < n - 1; k++) {
		if (t < x[k + 1]) {
			i  = k;
			break;
		}
	}
	if (i == -1) {
		i  = n - 2;
	}
	dd = t - x[i];
	dd2 = dd * dd; dd3 = dd2 * dd;

    return c[i] + c[n+i] * dd + c[2*n+i] * dd2 + c[3*n+i] * dd3;
}

double splineper3interpolate(int n, const double* c,
							 double t)
{
    double dd, dd2, dd3;
	int k, i = -1;
	for (k = 0; k < n; k++) {
		if (t < k + 1) {
			i  = k;
			break;
		}
	}
	if (i == -1) {
		//i  = n - 1;
		i  = n - 1;
	}
	dd = t - i;
	dd2 = dd * dd; dd3 = dd2 * dd;

    return c[i] + c[n+i] * dd + c[2*n+i] * dd2 + c[3*n+i] * dd3;
}
