# Run Examples

# Taylor Vortices
Run parameters:
```
./test/fdm_ns_cyl --plot:png=0 --plot:vtk=1 --plot:interval=1000 --ns:steps=10000 \
  --ns:Re=200 --ns:dt=0.01 --ns:h1=0 --ns:h2=10
```
Other defaults: r=pi/2, R=pi, nr=nz=nphi=32, u0=1

![Taylor](/img/taylor_200.png?raw=true)

# Cube box with moving lid
Run parameters:
```
./test/fdm_ns_cube --ns:nx=32 --ns:ny=32 --ns:nz=32 --plot:interval=1000 --ns:steps=10000 \
  --ns:Re=250 --ns:dt=0.01 --ns:u0=1 --plot:png=0 --plot:vtk=1
```
Other defaults: x1=y1=z1=-pi, x2=y2=z2=pi, u0=1

![Cube](/img/cube_250.png?raw=true)

# References
[1] L.D.Landau, E.M.Lifshitz, Fluid Mechanics, 2nd edition, 1987, Pergamon Press (оригинал: Л.Д.Ландау, Е.М.Лифшиц, Гидродинамика, издание 6-е исправленное, Москва, Физматлит, 2017)
[2] C.A.J.Fletcher, Computational Techniques for Fluid Dynamics, Springer, 1988 (перевод: К.Флетчер, Вычислительные методы в динамике жидкостей, Москва, Мир, 1991)
[3] A.A.Samarskii, E.S.Nikolaev, Numerical Methods For Grid Equations, Birkhauser Verlag, 1989 (оригинал: А.А.Самарский, Е.С.Николаев, Методы решения сеточных уравнений, Москва, Наука, 1978)
[4] J.W.Demmel, Applied Numerical Linear Algebra, SIAM, 1997 (перевод: Дж.Деммель, Вычислительная линейная алгбера, Москва, Мир, 2001)
[5] D.D.Joseph, Stability of fluid motions, Springer-Verlag, 1976 (перевод: Д.Джозеф, Устойчивость движений жидкости, Москва, Мир, 1981)
[6] Визуализация данных физического и математического моделирования в газовой динамике, под редакцией проф. В.Н.Емельянова, Москва, Физматлит, 2018
[7] Разностные схемы в задачах газовой динамики на неструктурированных сетках, под редакцией проф. В.Н.Емельянова, Москва, Физматлит, 2015
