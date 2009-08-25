#ifndef _MPPLUSPLUS_H
#define _MPPLUSPLUS_H
/*$Id$*/

/* Copyright (c) 2004, 2005 Alexey Ozeritsky
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
 * вычисление синуса и косинуса и экспоненты с точностью более 16 знаков
 */

#include <gmpxx.h>

typedef mpf_class mpf;

extern mpf *Pi;
extern mpf *Half_Pi;
extern mpf *Two_Pi;
extern mpf *E;
extern mpf *Eps;
extern mpf *Identity;
extern mpf *Ln10;
extern mpf *Ln2;


mpf sin(const mpf& r);
mpf cos(const mpf& r);
mpf exp(const mpf& t);
mpf pow(const mpf &t, int i);
mpf log(const mpf &t);
mpf log10(const mpf &t);

/*находит машинный нуль*/
mpf mp_machine_epsilon();
/**
 * устанавливает точность и инициализирует
 * математические константы
 */
void mp_init(unsigned long int prec);

#endif //_MPPLUSPLUS_H
