# fdm

# Вихри Тейлора
Параметры запуска
```
./test/fdm_ns_cyl --plot:png=0 --plot:vtk=1 --plot:interval=1000 --ns:steps=10000 \
  --ns:Re=200 --ns:dt=0.01 --ns:h1=0 --ns:h2=10
```
Параметры из кода: r=pi/2, R=pi, nr=nz=nphi=32, u0=1

![Taylor](/img/taylor_200.png?raw=true)
