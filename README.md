# Flux Reconstruction 2D Euler Solver (CUDA)

## Algorithm

1. **Mesh Generation**: Quadrilateral，Gauss-Lobatto-Legendre (3x3)
2. **Numerical Flux**: Lax-Friedrichs
3. **Correction Function**: g2
4. **Time Integration**: RK4
5. **Limiter**: TVB / Minmod limiter

## Requirements

- C++17 or higher
- CUDA Toolkit
- CMake

## Supported Case

- 2D Forward Facing Step (fws)
- 2D vortex in isentropic flow (inf)

## Build Instructions
   
```bash
mkdir build
cd build
cmake ..
make -j
./FR2DEulerSolver
```

## Configuration

`config.ini`
```ini
[Case]
type = fws

[Mesh]
width = 3.0
height = 1.0
nx = 30
ny = 10
fws_stepNX = 24
fws_stepNY = 2

[IC]
fws_rho = 1.4
fws_u = 3.0
fws_v = 0.0
fws_p = 1.0
inf_u = 1.0
inf_v = 1.0
inf_beta = 5.0

[BC]
fws_rho = 1.4
fws_u = 3.0
fws_v = 0.0
fws_p = 1.0

[Solver]
step_size = 0.00001
steps = 50000
method = RK4

[Limiter]
enable = true
M = 0.0

[Visiualize]
enable = true
gap = 1000
```

### Options:

- `type`: `fws` (Forward Facing Step) and `inf` (Isentropic Vortex).
- `fws_stepNX/fws_stepNY`: Number of cells in the step region for `fws`.
- `method`: Time integration method. `RK4` and `Euler` are supported.
