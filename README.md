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
