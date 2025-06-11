#ifndef EULER_EQ_CUH
#define EULER_EQ_CUH

#include "type_def.cuh"

#define _rho U[0]
#define _e U[3]
#define _u (U[1] / _rho)
#define _v (U[2] / _rho)
#define _p pressure(U, gamma)

__host__ __device__ inline double energy(const Vec4 &P, double gamma)
{
    return P[3] / (gamma - 1.0) + 0.5 * P[0] * (P[1] * P[1] + P[2] * P[2]);
}

__host__ __device__ inline double pressure(const Vec4 &U, double gamma)
{
    return (gamma - 1.0) * (_e - 0.5 * _rho * (_u * _u + _v * _v));
}

__host__ __device__ inline Vec4 toPrimitive(const Vec4 &U, double gamma)
{
    return {_rho, _u, _v, _p};
}

__host__ __device__ inline Vec4 toConservative(const Vec4 &P, double gamma)
{
    return {P[0], P[0] * P[1], P[0] * P[2], energy(P, gamma)};
}

__host__ __device__ inline Flux physicalFlux(const Vec4 &U, double gamma)
{
    Vec4 Fx, Fy;
    Fx[0] = _rho * _u;
    Fx[1] = _rho * _u * _u + _p;
    Fx[2] = _rho * _u * _v;
    Fx[3] = _u * (_e + _p);
    Fy[0] = _rho * _v;
    Fy[1] = _rho * _u * _v;
    Fy[2] = _rho * _v * _v + _p;
    Fy[3] = _v * (_e + _p);
    return {Fx, Fy};
}

#endif // EULER_EQ_CUH