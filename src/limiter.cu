#include "limiter.h"
#include "euler_eq.cuh"

struct Matrix4d
{
    double J[4][4];

    __device__ Vec4 operator*(const Vec4 &U) const
    {
        return {
            J[0][0] * U.x + J[0][1] * U.y + J[0][2] * U.z + J[0][3] * U.w,
            J[1][0] * U.x + J[1][1] * U.y + J[1][2] * U.z + J[1][3] * U.w,
            J[2][0] * U.x + J[2][1] * U.y + J[2][2] * U.z + J[2][3] * U.w,
            J[3][0] * U.x + J[3][1] * U.y + J[3][2] * U.z + J[3][3] * U.w};
    }
};

__device__ double minmod(double a, double b, double c)
{
    if (a * b <= 0 || a * c <= 0 || b * c <= 0)
    {
        return 0.0;
    }
    return fmin(fabs(a), fmin(fabs(b), fabs(c))) * (a > 0 ? 1 : -1);
}

__device__ double tvb(double a, double b, double c, double Mh2)
{
    double minmod_value = minmod(a, b, c);
    if (minmod_value == 0.0)
    {
        return 0.0;
    }
    double result = fmin(fabs(minmod_value), Mh2);
    return minmod_value > 0 ? result : -result;
}

__device__ void eigenDecomposeX(const Vec4 &U, Matrix4d &RX, Matrix4d &RinvX, const double gamma)
{
    const double c = sqrt(gamma * _p / _rho);
    const double H = (_e + _p) / _rho;

    RX.J[0][0] = 1.0;
    RX.J[1][0] = _u - c;
    RX.J[2][0] = _v;
    RX.J[3][0] = H - _u * c;

    RX.J[0][1] = 1.0;
    RX.J[1][1] = _u;
    RX.J[2][1] = _v;
    RX.J[3][1] = 0.5 * (_u * _u + _v * _v);

    RX.J[0][2] = 0.0;
    RX.J[1][2] = 0.0;
    RX.J[2][2] = 1.0;
    RX.J[3][2] = _v;

    RX.J[0][3] = 1.0;
    RX.J[1][3] = _u + c;
    RX.J[2][3] = _v;
    RX.J[3][3] = H + _u * c;

    const double inv_c = 1.0 / c;
    const double inv_c2 = inv_c * inv_c;
    const double gm1 = gamma - 1.0;

    RinvX.J[0][0] = gm1 * (_u * _u + _v * _v) / (2.0 * c * c) + _u * inv_c;
    RinvX.J[0][1] = -gm1 * _u * inv_c2 - inv_c;
    RinvX.J[0][2] = -gm1 * _v * inv_c2;
    RinvX.J[0][3] = gm1 * inv_c2;

    RinvX.J[1][0] = 1.0 - gm1 * (_u * _u + _v * _v) * inv_c2;
    RinvX.J[1][1] = gm1 * _u * inv_c2;
    RinvX.J[1][2] = gm1 * _v * inv_c2;
    RinvX.J[1][3] = -gm1 * inv_c2;

    RinvX.J[2][0] = -_v;
    RinvX.J[2][1] = 0.0;
    RinvX.J[2][2] = 1.0;
    RinvX.J[2][3] = 0.0;

    RinvX.J[3][0] = gm1 * (_u * _u + _v * _v) / (2.0 * c * c) - _u * inv_c;
    RinvX.J[3][1] = -gm1 * _u * inv_c2 + inv_c;
    RinvX.J[3][2] = -gm1 * _v * inv_c2;
    RinvX.J[3][3] = gm1 * inv_c2;
}

__device__ void eigenDecomposeY(const Vec4 &U, Matrix4d &RY, Matrix4d &RinvY, const double gamma)
{
    const double c = sqrt(gamma * _p / _rho);
    const double H = (_e + _p) / _rho;

    RY.J[0][0] = 1.0;
    RY.J[1][0] = _v - c;
    RY.J[2][0] = _u;
    RY.J[3][0] = H - _v * c;

    RY.J[0][1] = 1.0;
    RY.J[1][1] = _v;
    RY.J[2][1] = _u;
    RY.J[3][1] = 0.5 * (_v * _v + _u * _u);

    RY.J[0][2] = 0.0;
    RY.J[1][2] = 1.0;
    RY.J[2][2] = 0.0;
    RY.J[3][2] = _u;

    RY.J[0][3] = 1.0;
    RY.J[1][3] = _v + c;
    RY.J[2][3] = _u;
    RY.J[3][3] = H + _v * c;

    const double inv_c = 1.0 / c;
    const double inv_c2 = inv_c * inv_c;
    const double gm1 = gamma - 1.0;

    RinvY.J[0][0] = gm1 * (_v * _v + _u * _u) / (2.0 * c * c) + _v * inv_c;
    RinvY.J[0][1] = -gm1 * _v * inv_c2 - inv_c;
    RinvY.J[0][2] = -gm1 * _u * inv_c2;
    RinvY.J[0][3] = gm1 * inv_c2;

    RinvY.J[1][0] = 1.0 - gm1 * (_v * _v + _u * _u) * inv_c2;
    RinvY.J[1][1] = gm1 * _v * inv_c2;
    RinvY.J[1][2] = gm1 * _u * inv_c2;
    RinvY.J[1][3] = -gm1 * inv_c2;

    RinvY.J[2][0] = -_u;
    RinvY.J[2][1] = 1.0;
    RinvY.J[2][2] = 0.0;
    RinvY.J[2][3] = 0.0;

    RinvY.J[3][0] = gm1 * (_v * _v + _u * _u) / (2.0 * c * c) - _v * inv_c;
    RinvY.J[3][1] = -gm1 * _v * inv_c2 + inv_c;
    RinvY.J[3][2] = -gm1 * _u * inv_c2;
    RinvY.J[3][3] = gm1 * inv_c2;
}

__device__ __forceinline__ Vec4 cell_mean(const Vec4 *nodes)
{
    constexpr double gll_weight_2d[] = {
        1.0 / 36.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 9.0,
        4.0 / 36.0};
    Vec4 mean{};
    for (int i = 0; i < 9; ++i)
    {
        mean += nodes[i] * gll_weight_2d[i];
    }
    return mean;
}

__device__ __forceinline__ Vec4 face_mean(const Vec4 &left, const Vec4 &center, const Vec4 &right)
{
    return left / 6.0 + center * 2.0 / 3.0 + right / 6.0;
}

__device__ void limiter_1d(
    const Vec4 &left_mean,
    const Vec4 &local_mean,
    const Vec4 &right_mean,
    const Vec4 &local_left,
    const Vec4 &local_right,
    const Matrix4d &R,
    const Matrix4d &Rinv,
    Vec4 &new_local_left,
    Vec4 &new_local_right,
    const double gamma,
    const double Mh2 = 0.0)
{
    const Vec4 left_mean_V = Rinv * left_mean;
    const Vec4 local_mean_V = Rinv * local_mean;
    const Vec4 right_mean_V = Rinv * right_mean;
    const Vec4 local_left_V = Rinv * local_left;
    const Vec4 local_right_V = Rinv * local_right;

    const Vec4 delta_left_V = local_mean_V - local_left_V;
    const Vec4 delta_right_V = local_right_V - local_mean_V;
    const Vec4 D_left_V = local_mean_V - left_mean_V;
    const Vec4 D_right_V = right_mean_V - local_mean_V;

    Vec4 delta_left_mod_V;
    Vec4 delta_right_mod_V;
    for (int i = 0; i < 4; ++i)
    {
        delta_left_mod_V[i] = tvb(delta_left_V[i], D_left_V[i], D_right_V[i], Mh2);
        delta_right_mod_V[i] = tvb(delta_right_V[i], D_left_V[i], D_right_V[i], Mh2);
    }
    new_local_left += (local_mean - R * delta_left_mod_V) * 0.5;
    new_local_right += (local_mean + R * delta_right_mod_V) * 0.5;
}

__device__ __forceinline__ int idx(const int i, const int j)
{
    const int id = j * 3 + i;
    constexpr int _idx[] = {0, 1, 2, 7, 8, 3, 6, 5, 4};
    return _idx[id];
}

__device__ __forceinline__ void set_bc(const Vec4 &bc, const Vec4 *local_U, const int bc_type, Vec4 *neighbour_U,
                                       const double gamma)
{
    for (int j = 0; j < 3; ++j)
    {
        if (bc_type == X_WALL)
        {
            neighbour_U[j] = Vec4(local_U[j][0], -local_U[j][1], local_U[j][2], local_U[j][3]);
        }
        else if (bc_type == Y_WALL)
        {
            neighbour_U[j] = Vec4(local_U[j][0], local_U[j][1], -local_U[j][2], local_U[j][3]);
        }
        else if (bc_type == INLET)
        {
            neighbour_U[j] = bc;
        }
        else
        {
            neighbour_U[j] = local_U[j];
        }
    }
}

__global__ void limiter_kernel(
    const Cell *d_cells,
    const Face *d_faces,
    const Vec4 *d_nodes,
    Vec4 *d_new_nodes,
    const int num_elements,
    const Vec4 bc_Q,
    const double gamma,
    const double Mh2 = 0)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements)
    {
        return;
    }
    const Cell &cell = d_cells[i];
    if (!cell.isValid)
    {
        return;
    }
    const Vec4 *node = d_nodes + i * 9;
    Vec4 *new_node = d_new_nodes + i * 9;
    for (int j = 0; j < 9; ++j)
    {
        new_node[j] = Vec4();
    }
    const Vec4 U_mean = cell_mean(node);
    Matrix4d RX, RinvX, RY, RinvY;
    eigenDecomposeX(U_mean, RX, RinvX, gamma);
    eigenDecomposeY(U_mean, RY, RinvY, gamma);

    Vec4 local_mean_X[3], local_mean_Y[3];
    for (int j = 0; j < 3; ++j)
    {
        local_mean_X[j] = face_mean(node[idx(0, j)], node[idx(1, j)], node[idx(2, j)]);
        local_mean_Y[j] = face_mean(node[idx(j, 0)], node[idx(j, 1)], node[idx(j, 2)]);
    }

    Vec4 left_mean[3], right_mean[3], bottom_mean[3], top_mean[3];

    const Face &bottom_face = d_faces[cell.faceIds[0]];
    const Face &right_face = d_faces[cell.faceIds[1]];
    const Face &top_face = d_faces[cell.faceIds[2]];
    const Face &left_face = d_faces[cell.faceIds[3]];

    // bottom cell
    if (bottom_face.leftCell < 0)
    {
        set_bc(bc_Q, local_mean_Y, bottom_face.leftCell, bottom_mean, gamma);
    }
    else
    {
        const Vec4 *bottom_node = d_nodes + bottom_face.leftCell * 9;
        for (int j = 0; j < 3; ++j)
        {
            bottom_mean[j] = face_mean(bottom_node[idx(j, 0)], bottom_node[idx(j, 1)], bottom_node[idx(j, 2)]);
        }
    }

    // right cell
    if (right_face.rightCell < 0)
    {
        set_bc(bc_Q, local_mean_X, right_face.rightCell, right_mean, gamma);
    }
    else
    {
        const Vec4 *right_node = d_nodes + right_face.rightCell * 9;
        for (int j = 0; j < 3; ++j)
        {
            right_mean[j] = face_mean(right_node[idx(0, j)], right_node[idx(1, j)], right_node[idx(2, j)]);
        }
    }

    // top cell
    if (top_face.rightCell < 0)
    {
        set_bc(bc_Q, local_mean_Y, top_face.rightCell, top_mean, gamma);
    }
    else
    {
        const Vec4 *top_node = d_nodes + top_face.rightCell * 9;
        for (int j = 0; j < 3; ++j)
        {
            top_mean[j] = face_mean(top_node[idx(j, 0)], top_node[idx(j, 0)], top_node[idx(j, 0)]);
        }
    }

    // left cell
    if (left_face.leftCell < 0)
    {
        set_bc(bc_Q, local_mean_X, left_face.leftCell, left_mean, gamma);
    }
    else
    {
        const Vec4 *left_node = d_nodes + left_face.leftCell * 9;
        for (int j = 0; j < 3; ++j)
        {
            left_mean[j] = face_mean(left_node[idx(0, j)], left_node[idx(1, j)], left_node[idx(2, j)]);
        }
    }

    for (int j = 0; j < 3; ++j)
    {
        // X
        new_node[idx(1, j)] += local_mean_X[j] * 0.5;
        limiter_1d(
            left_mean[j],
            local_mean_X[j],
            right_mean[j],
            node[idx(0, j)],
            node[idx(2, j)],
            RX,
            RinvX,
            new_node[idx(0, j)],
            new_node[idx(2, j)],
            gamma,
            Mh2);
        // Y
        new_node[idx(j, 1)] += local_mean_Y[j] * 0.5;
        limiter_1d(
            bottom_mean[j],
            local_mean_Y[j],
            top_mean[j],
            node[idx(j, 0)],
            node[idx(j, 2)],
            RY,
            RinvY,
            new_node[idx(j, 0)],
            new_node[idx(j, 2)],
            gamma,
            Mh2);
    }
}

void tvd_limiter(
    const Cell *d_cells,
    const Face *d_faces,
    const Vec4 *d_nodes,
    Vec4 *d_new_nodes,
    const int num_elements,
    const Vec4 &bc_Q,
    const double gamma,
    const double Mh2)
{
    const int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    limiter_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_cells, d_faces, d_nodes, d_new_nodes, num_elements, bc_Q, gamma, Mh2);
    cudaDeviceSynchronize();
}
