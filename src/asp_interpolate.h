#ifndef _ASP_INTERPOLATE_H
#define _ASP_INTERPOLATE_H
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
#ifdef __cplusplus
extern "C" {
#endif
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

void bicubic_resample(int oldwidth,
     int oldheight,
     int newwidth,
     int newheight,
     const double * a,
     double * b);

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
     double * b);

void spline3buildtable(int n,
					   double *x,
					   double * y,
					   double * C);

/*для периодического сплайна
нулевая и последняя точки совпадают в массивах
размерность массивов n + 1
*/
void splineper3buildtable(int n,
						  double * y,
						  double * C);

double spline3interpolate(int n, const double* c, const double *x,
						  double t);

/*для периодического сплайна*/
double splineper3interpolate(int n, const double* c,
							 double t);

#ifdef __cplusplus
} /*extern "C" {*/
#endif
#endif /*_ASP_INTERPOLATE_H*/
