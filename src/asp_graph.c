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

#include <stdlib.h>
#include <string.h>
#include "asp_graph.h"

#ifndef sign
#define sign(a) ((a)>0)?1:-1
#endif

void setpixel(void * data, int w, int h, 
			  int x, int y, void * color, size_t sz)
{
	void * ptr;
	if (x < 0 || x > w - 1) return;
	if (y < 0 || y > h - 1) return;
	ptr = (char*)data + (y * w + x) * sz;
	memcpy(ptr, color, sz);
}

void line(void * data, int w, int h, int x1, int y1, int x2, int y2,
		  void * color, size_t sz)
{
    int x, y;
    int dx, dy;
    int sx, sy;
    int t, e;
    int change;
	int i;

    x = x1;
    y = y1;

    dx = abs(x2 - x1);
    dy = abs(y2 - y1);
    sx = sign(x2 - x1);
    sy = sign(y2 - y1);
    change = 0;
    if (dx < dy) {
        //надо двигаться по Y, посему меняем координаты
        t = dx; dx = dy; dy = t;
        change = 1;
    }
    e = 2 * dy - dx;

    for (i = 0; i < dx; i++) {
		setpixel(data, w, h, x, y, color, sz);

        while (e >= 0) {
            if (change) {
                x = x + sx;
            } else {
                y = y + sy;
            }
            e = e - 2 * dx;
        }
        if (change) {
            y = y + sy;
        } else {
            x = x + sx;
        }
        e = e + 2 * dy;
    }
	setpixel(data, w, h, x2, y2, color, sz);
}

void circle(void * data, int w, int h, int xc, int yc, int r,
		  void * color, size_t sz)
{
    int x, y;
    int d;

    x = 0;
    y = r;
    d = 3 - 2 * r;

    while (y >= x) {
        setpixel(data, w, h, x+xc, y+yc, color, sz);
        setpixel(data, w, h, x+xc, -y+yc, color, sz);
        setpixel(data, w, h, -x+xc, y+yc, color, sz);
        setpixel(data, w, h, -x+xc, -y+yc, color, sz);
        setpixel(data, w, h, y+xc, x+yc, color, sz);
        setpixel(data, w, h, y+xc, -x+yc, color, sz);
        setpixel(data, w, h, -y+xc, x+yc, color, sz);
        setpixel(data, w, h, -y+xc, -x+yc, color, sz);
        if (d < 0) {
            d = d + 4 * x + 6;
        } else {
            d = d + 4 * (x - y) + 10;
            y = y - 1;
        }
        x = x + 1;
    }
}

stack * stack_alloc(int _size, size_t esz)
{
	stack * st = (stack*) malloc(_size * sizeof(stack));
	st->esz  = esz;
	st->data = malloc(esz * _size);
	st->size = _size;
	st->ptr  = -1;
	return st;
}

void stack_free(stack * st)
{
	free(st->data);
	free(st);
}

int stack_empty(stack * st)
{
	if (st->ptr < 0) return 1;
	else return 0;
}

int stack_full(stack * st)
{
	if (st->size - 1 < st->ptr)
		return 1;
	else return 0;
}

void * stack_pop(stack * st)
{
	if (!stack_empty(st)) {
		void * ret = (char*)st->data + st->ptr * st->esz;
		st->ptr --;
		return ret;
	} else {
		return 0;
	}
}

void stack_push(stack * st, void * data)
{
	if (!stack_full(st)) {
		void * ptr;
		st->ptr ++;
		ptr = (char*)st->data + st->ptr * st->esz;
		memcpy(ptr, data, st->esz);
	}
}

int testpixel(void * data, int w, int h, int x, int y, void * color,
					 size_t sz)
{
	void * ptr = (char*)data + (y * w + x) * sz;
	return !memcmp(ptr, color, sz);
}

void * getpixel(void * data, int w, int h, int x, int y, size_t sz)
{
	void * ptr = (char*)data + (y * w + x) * sz;
	return ptr;
}

/*ncolor - новый цвет, gcolor - цвет границы*/
void fill(void * data, int w, int h, 
		  int sx, int sy, void * ncolor, void * gcolor, size_t sz)
{
	stack * stk_x = stack_alloc(w * h, sizeof(int));
	stack * stk_y = stack_alloc(w * h, sizeof(int));

	int x, y;
	int * x_p, *y_p;

	stack_push(stk_x, &sx);
	stack_push(stk_y, &sy);

	while (!stack_empty(stk_x)) {    /* Пока не исчерпан стек */
		/* Выбираем пиксел из стека и красим его */
		x_p = stack_pop(stk_x);
		y_p = stack_pop(stk_y);
		x = *x_p;
		y = *y_p;

		if ((x < 0 || x > w) || (y < 0 || y > h)) {
			continue;
		}

		if (!testpixel(data, w, h, x, y, gcolor, sz)) {
			setpixel(data, w, h, x, y, ncolor, sz);
		} else {
			continue;
		}

		/*право*/
		if (x < w - 1) {
			int nx = x + 1;
			if (!testpixel(data, w, h, x + 1, y, ncolor, sz)) {
				stack_push(stk_x, &nx);
				stack_push(stk_y, &y);
			}
		}

		/*верх*/
		if (y < h - 1) {
			int ny = y + 1;
			if (!testpixel(data, w, h, x, y + 1, ncolor, sz)) {
				stack_push(stk_x, &x);
				stack_push(stk_y, &ny);
			}
		}

		/*лево*/
		if (x > 0) {
			int nx = x - 1;
			if (!testpixel(data, w, h, x - 1, y, ncolor, sz)) {
				stack_push(stk_x, &nx);
				stack_push(stk_y, &y);
			}
		}

		/*низ*/
		if (y > 0) {
			int ny = y - 1;
			if (!testpixel(data, w, h, x, y - 1, ncolor, sz)) {
				stack_push(stk_x, &x);
				stack_push(stk_y, &ny);
			}
		}
	}

	stack_free(stk_x); 
	stack_free(stk_y);
}
