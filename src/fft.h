#pragma once

#include <vector>

namespace fdm {

template<typename T>
class FFTTable {
public:
    int N;
    std::vector<T> ffCOS;
    std::vector<T> ffSIN;
    std::vector<T> ffiCOS;

    FFTTable(int N): N(N)
    {
        init();
    }

    T iCOS(int k, int l) const {
        return ffiCOS[(l+1)*N+(k-1)];
    }

private:
    void init();
};

template<typename T>
class FFT {
    const FFTTable<T>& t;

    int N;
    int n; // N = 2^n

public:
    FFT(const FFTTable<T>& table, int N)
        : t(table)
        , N(N)
    {
        init();
    }

    /*!быстрое преобразование Фурье периодической функции.
      по коэфф Фурье находим значения функции.
      fk->f(i)
      Самарский-Николаев, страница 180-181
      \param S  - ответ
      \param s  - начальное условие
      \param dx - множитель перед суммой
	*/
    void pFFT(T *S, T* s, T dx);

	/*!быстрое преобразование Фурье периодической функции.
      по значениям функции находим коэфф Фурье.
      f(i)->fk
      Самарский-Николаев, страница 180-181, формулы 65-66
      \param S  - ответ
      \param s  - начальное условие
      \param dx - множитель перед суммой
	*/
	void pFFT_1_old(T *S, T *s1, T dx);
	/*! быстрое косинусное преобразование.
	   Самарский-Николаев, страница 176, формулы 46-47
       fftw: REDFT00
       S и s: массивы размера N+1, 0-indexing
	 */
    void cFFT_old(T *S, T *s, T dx);
	/*! быстрое синусное преобразование.
	   Самарский-Николаев, страница 180
       fftw: RODFT00
       S и s: массивы размера N-1, 1-indexing
	 */
    void sFFT_old(T *S, T *s, T dx);

    // new optimized version
    void sFFT(T* S, T* s, T dx);
    void cFFT(T* S, T* s, T dx);
    void pFFT_1(T* S, T* s, T dx);

private:
    void init();

    void sFFT_old(T *S, T *s, T dx, int N, int n,int nr);
    void cFFT_old(T *S, T *s, T dx, int N, int n,int nr);

    void sFFT(T* S, T* s, T dx, int N, int n);
    void cFFT(T* S, T* s, T dx, int N, int n);

    inline void padvance(T*a, int idx);
    inline void cadvance(T*a, int idx);
    inline void sadvance(T*a, int idx);
};

} // namespace fdm
