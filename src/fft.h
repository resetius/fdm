#pragma once

#include <vector>

namespace fdm {

template<typename T>
class FFTTable {
public:
    int N;
    std::vector<T> ffCOS;
    std::vector<T> ffSIN;

    FFTTable(int N)
    {
        init(N);
    }

private:
    void init(int N);
};

template<typename T>
class FFT {
    const FFTTable<T>& t;

    int N;
    int n; // N = 2^n

    std::vector<T> a;
    std::vector<T> y;
    std::vector<T> _y;

    std::vector<T> ss;

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
	void pFFT_1(T *S, T *s1, T dx);
	/*! быстрое косинусное преобразование.
	   Самарский-Николаев, страница 176, формулы 46-47
       fftw: REDFT00
       S и s: массивы размера N+1, 0-indexing
	 */
    void cFFT(T *S, T *s, T dx);
	/*! быстрое синусное преобразование.
	   Самарский-Николаев, страница 180
       fftw: RODFT00
       S и s: массивы размера N-1, 1-indexing
	 */
    void sFFT(T *S, T *s, T dx);

private:
    void init();

    void sFFT(T *S, T *s, T dx, int N, int n,int nr);
    void cFFT(T *S, T *s, T dx, int N, int n,int nr);

    inline void padvance(T*a, int idx);
};

} // namespace fdm
