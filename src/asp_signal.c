/*$Id$*/
/* Copyright (c) 2006 Alexey Ozeritsky
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

/**
 * Обработка сигналов
 */

#include <stdio.h>
#include <stdlib.h>
#include "asp_signal.h"

sig_counters_t sig_cnt = {
	/*.sigint  =*/ 0,
	/*.sigusr1 =*/ 0,
	/*.sigusr2 =*/ 0,
};

#if defined(_HAS_SIGNALS)
/*надежная функция signal()*/
/*пример из Робачевского   */
void (*mysignal (int signo, void (*hndlr)(int)))(int)
{
	struct sigaction act,oact;
	/*установим маску сигналов*/
	act.sa_handler = hndlr;
	sigemptyset(&act.sa_mask);
	act.sa_flags = 0;
	if(signo != SIGALRM)
		act.sa_flags |= SA_RESTART;
	/*установим диспозицию*/
	if(sigaction(signo,&act,&oact)<0)
		return(SIG_ERR);
	return(oact.sa_handler);
}
#else
void (*mysignal (int signo, void (*hndlr)(int)))(int)
{
	return 0;
}
#endif

/*обработчик сигнала SIGINT*/
void sigint_hndlr(int signo)
{
	if (sig_cnt.sigint > 0)
	{
		fprintf(stderr,"Interrupt. Exiting ... \n");
		exit(1);
	}
	else
		fprintf(stderr,"Dumping current result ....\n");
	sig_cnt.sigint=1;
}

/*обработчик сигнала SIGUSR*/
void sigusr_hndlr(int signo)
{
#if defined(_HAS_SIGNALS)
	fprintf(stderr, "SIGUSR catched\n");
	if (signo == SIGUSR1) {
		++sig_cnt.sigusr1;
	} else if (signo == SIGUSR2) {
		++sig_cnt.sigusr2;
	}
#endif
}
