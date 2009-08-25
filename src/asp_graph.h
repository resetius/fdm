#ifndef _ASP_GRAPH_H
#define _ASP_GRAPH_H
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
/* функции работы с растровыми данными */

#ifdef __cplusplus
extern "C" {
#endif
void line(void * data, int w, int h, int x1, int y1, int x2, int y2,
		  void * color, size_t sz);

void circle(void * data, int w, int h, int x, int y, int r,
		  void * color, size_t sz);

void setpixel(void * data, int w, int h, 
			  int x, int y, void * color, size_t sz);
void * getpixel(void * data, int w, int h, int x, int y, size_t sz);

/*проверяет пиксел на сетке на равенство значению color*/
int testpixel(void * data, int w, int h, int x, int y, void * color,
			  size_t sz);
/*ncolor - новый цвет, gcolor - цвет границы*/
void fill(void * data, int w, int h, 
		  int x, int y, void * ncolor, void * gcolor, size_t sz);

/*стек*/
typedef struct _stack{
	void * data;
	size_t esz; /*!<размер элемента*/
	int size; /*!<максимальное число элементов*/
	int ptr;  /*!<указатель на текущий элемент*/
} stack;

void stack_push(stack * st, void * data);
void * stack_pop(stack * st);
int stack_full(stack * st);
int stack_empty(stack * st);
stack * stack_alloc(int _size, size_t esz);
void stack_free(stack *);
#ifdef __cplusplus
}
#endif
#endif //_ASP_GRAPH_H
