/*  $Id$   */

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
 * Мелкие функции с произвольной точностью.
 */

#include <math.h>
#include <gmp.h>
#include <gmpxx.h>
#include "asp_lib.h"
#include "asp_misc_mp.h"

mpf_class dist(mpf_class *y1, mpf_class *y2,int n)
{
	int i;
	mpf_class n2=0;
	for(i=0;i<n;i++)
	{
		n2+=(y1[i]-y2[i])*(y1[i]-y2[i]);
	}
	n2=sqrt(n2);
	return n2;
}


mpf_class max3(mpf_class x, mpf_class y, mpf_class z)
{
	if(x>y&&x>z)
		return x;
	else if(y>x&&y>z)
		return y;
	else
		return z;
}

mpf_class norm1(mpf_class *x,int n)
{
	int i;
	mpf_class n1=0.0;
	mpf_class f;
	for(i=0;i<n;i++)
	{
		f=abs(x[i]);
		if(n1<f) n1=f;
	}
	return n1;
}

mpf_class norm2(mpf_class *x,int n)
{
	int i;
	mpf_class n2=0.0;
	for(i=0;i<n;i++)
		n2+=x[i]*x[i];
	n2=sqrt(n2);
	return n2;
}

