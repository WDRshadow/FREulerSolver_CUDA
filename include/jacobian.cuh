#ifndef JACOBIAN_H
#define JACOBIAN_H

#include <cmath>

#include "type_def.cuh"

struct Matrix2d
{
    double J[2][2];

    __host__ __device__ double det() const
    {
        return J[0][0] * J[1][1] - J[0][1] * J[1][0];
    }

    __host__ __device__ Matrix2d inv() const
    {
        Matrix2d J_inv{};
        double detJ = det();
        J_inv.J[0][0] = J[1][1] / detJ;
        J_inv.J[0][1] = -J[0][1] / detJ;
        J_inv.J[1][0] = -J[1][0] / detJ;
        J_inv.J[1][1] = J[0][0] / detJ;
        return J_inv;
    }

    __host__ __device__ Matrix2d T() const
    {
        Matrix2d J_T{};
        J_T.J[0][0] = J[0][0];
        J_T.J[0][1] = J[1][0];
        J_T.J[1][0] = J[0][1];
        J_T.J[1][1] = J[1][1];
        return J_T;
    }

    __host__ __device__ Flux operator*(const Flux &b) const
    {
        Flux result;
        result.f = b.f * J[0][0] + b.g * J[0][1];
        result.g = b.f * J[1][0] + b.g * J[1][1];
        return result;
    }
};

void calculate_jacobian_face(const Mesh &mesh, double *mesh_jacobian_face);
void calculate_jacobian_cell(const Mesh &mesh, Matrix2d *mesh_jacobian_cell_invT, double *mesh_jacobian_cell_det);

#endif // JACOBIAN_H