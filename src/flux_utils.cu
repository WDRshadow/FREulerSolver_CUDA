#include "flux_utils.h"
#include "euler_eq.cuh"
#include "jacobian.cuh"

#define BLOCK_SIZE 256

__global__ void physical_flux_kernel(const Vec4 *d_nodes, Flux *d_fluxs, const int num_elements, const double gamma)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements)
    {
        return;
    }
    const Vec4 *node = d_nodes + i * 9;
    Flux *flux = d_fluxs + i * 9;
    for (int j = 0; j < 9; ++j)
    {
        const Vec4 &Q = node[j];
        flux[j] = physicalFlux(Q, gamma);
    }
}

void calculate_physical_flux(const Vec4 *d_nodes, Flux *d_fluxs, const int num_elements, const double gamma)
{
    const int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    physical_flux_kernel<<<num_blocks, BLOCK_SIZE>>>(d_nodes, d_fluxs, num_elements, gamma);
    cudaDeviceSynchronize();
}

__device__ __forceinline__ double maxEig(const Vec4 &Q, const Point &normal, const double gamma)
{
    const Vec4 P = toPrimitive(Q, gamma);
    const double nx = normal.x;
    const double ny = normal.y;
    const double un = P[1] * nx + P[2] * ny;
    const double c = sqrt(gamma * P[3] / P[0]);
    return fabs(un) + c;
}

__device__ __forceinline__ Vec4 l_f_flux(
    const Vec4 &QL,
    const Flux &FL,
    const Vec4 &QR,
    const Flux &FR,
    const Point &normal,
    const double gamma)
{
    const double nx = normal.x, ny = normal.y;
    const double sL = maxEig(QL, normal, gamma);
    const double sR = maxEig(QR, normal, gamma);
    const double s = max(sL, sR);
    return (FL.f * nx + FL.g * ny + FR.f * nx + FR.g * ny) * 0.5 - (QR - QL) * 0.5 * s;
}

__device__ __forceinline__ void set_bc(const Vec4 &bc, const Vec4 *Q, const int bc_type, Vec4 *nodes, Flux *face_flux,
                                       const double gamma)
{
    for (int j = 0; j < 3; ++j)
    {
        if (bc_type == X_WALL)
        {
            nodes[j] = Vec4(Q[j][0], -Q[j][1], Q[j][2], Q[j][3]);
        }
        else if (bc_type == Y_WALL)
        {
            nodes[j] = Vec4(Q[j][0], Q[j][1], -Q[j][2], Q[j][3]);
        }
        else if (bc_type == INLET)
        {
            nodes[j] = bc;
        }
        else
        {
            nodes[j] = Q[j];
        }
        face_flux[j] = physicalFlux(nodes[j], gamma);
    }
}

__device__ __forceinline__ void set_nodes(
    const Vec4 *node,
    const Flux *flux,
    Vec4 *face_node,
    Flux *face_flux,
    const int idx[3])
{
    for (int j = 0; j < 3; ++j)
    {
        face_node[j] = node[idx[j]];
        face_flux[j] = flux[idx[j]];
    }
}

__global__ void face_num_flux_kernel(
    const Face *d_faces,
    const Vec4 *d_nodes,
    const Flux *d_fluxs,
    Vec4 *d_face_fluxs,
    const int nx,
    const int ny,
    const Vec4 bc,
    const double gamma)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int numVerticalFace = (nx + 1) * ny;
    const int numHorizontalFace = nx * (ny + 1);
    if (i >= numVerticalFace + numHorizontalFace)
    {
        return;
    }
    const Face &face = d_faces[i];
    const Point &normal = face.normal;
    constexpr int left_idx[] = {0, 7, 6};
    constexpr int right_idx[] = {2, 3, 4};
    constexpr int bottom_idx[] = {0, 1, 2};
    constexpr int top_idx[] = {6, 5, 4};
    if (face.leftCell < 0 && face.rightCell < 0)
    {
        return;
    }
    Vec4 l_nodes[3], r_nodes[3];
    Flux l_fluxs[3], r_fluxs[3];
    if (face.leftCell >= 0)
    {
        const Vec4 *left_node = d_nodes + face.leftCell * 9;
        const Flux *left_flux = d_fluxs + face.leftCell * 9;
        if (i < numVerticalFace)
        {
            set_nodes(left_node, left_flux, l_nodes, l_fluxs, right_idx);
        }
        else
        {
            set_nodes(left_node, left_flux, l_nodes, l_fluxs, top_idx);
        }
    }
    if (face.rightCell >= 0)
    {
        const Vec4 *right_node = d_nodes + face.rightCell * 9;
        const Flux *right_flux = d_fluxs + face.rightCell * 9;
        if (i < numVerticalFace)
        {
            set_nodes(right_node, right_flux, r_nodes, r_fluxs, left_idx);
        }
        else
        {
            set_nodes(right_node, right_flux, r_nodes, r_fluxs, bottom_idx);
        }
    }
    if (face.leftCell < 0)
    {
        set_bc(bc, r_nodes, face.leftCell, l_nodes, l_fluxs, gamma);
    }
    if (face.rightCell < 0)
    {
        set_bc(bc, l_nodes, face.rightCell, r_nodes, r_fluxs, gamma);
    }
    for (int j = 0; j < 3; ++j)
    {
        d_face_fluxs[i * 3 + j] = l_f_flux(l_nodes[j], l_fluxs[j], r_nodes[j], r_fluxs[j], normal, gamma);
    }
}

void calculate_face_num_flux(
    const Face *d_faces,
    const Vec4 *d_nodes,
    const Flux *d_fluxs,
    Vec4 *d_face_fluxs,
    const Vec4 &bc_Q,
    const int nx,
    const int ny,
    const double gamma)
{
    const int numFaces = (nx + 1) * ny + nx * (ny + 1);
    const int num_blocks = (numFaces + BLOCK_SIZE - 1) / BLOCK_SIZE;
    face_num_flux_kernel<<<num_blocks, BLOCK_SIZE>>>(d_faces, d_nodes, d_fluxs, d_face_fluxs, nx, ny, bc_Q, gamma);
    cudaDeviceSynchronize();
}

__device__ __forceinline__ double l_1d(const int i, const double s)
{
    switch (i)
    {
    case 0:
        return s * (s - 1.0) / 2.0;
    case 1:
        return 1.0 - s * s;
    case 2:
        return s * (s + 1.0) / 2.0;
    default:
        return 0.0;
    }
}

__device__ __forceinline__ double dl_1d(const int i, const double s)
{
    switch (i)
    {
    case 0:
        return s - 0.5;
    case 1:
        return -2.0 * s;
    case 2:
        return s + 0.5;
    default:
        return 0.0;
    }
}

__device__ __forceinline__ int idx_xi(const int i)
{
    constexpr int idx_xi_[] = {-1, 0, 1, 1, 1, 0, -1, -1, 0};
    return idx_xi_[i];
}

__device__ __forceinline__ int idx_eta(const int i)
{
    constexpr int idx_eta_[] = {-1, -1, -1, 0, 1, 1, 1, 0, 0};
    return idx_eta_[i];
}

__device__ __forceinline__ double l_2d(const int i, const double xi, const double eta)
{
    return l_1d(idx_xi(i) + 1, xi) * l_1d(idx_eta(i) + 1, eta);
}

__device__ __forceinline__ double dl_dxi(const int i, const double xi, const double eta)
{
    return dl_1d(idx_xi(i) + 1, xi) * l_1d(idx_eta(i) + 1, eta);
}

__device__ __forceinline__ double dl_deta(const int i, const double xi, const double eta)
{
    return l_1d(idx_xi(i) + 1, xi) * dl_1d(idx_eta(i) + 1, eta);
}

__global__ void div_flux_kernel(
    const Flux *d_fluxs,
    Flux *d_dfluxs,
    const int num_elements)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements)
    {
        return;
    }
    const Flux *flux = d_fluxs + i * 9;
    Flux *dflux = d_dfluxs + i * 9;
    for (int _i = 0; _i < 9; ++_i)
    {
        dflux[_i] = Flux();
        for (int _j = 0; _j < 9; ++_j)
        {
            const double dl_dxi_ = dl_dxi(_j, idx_xi(_i), idx_eta(_i));
            const double dl_deta_ = dl_deta(_j, idx_xi(_i), idx_eta(_i));
            dflux[_i].f += flux[_j].f * dl_dxi_;
            dflux[_i].g += flux[_j].g * dl_deta_;
        }
    }
}

void calculate_physical_div_flux(
    const Flux *d_fluxs,
    Flux *d_dfluxs,
    const int num_elements)
{
    const int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    div_flux_kernel<<<num_blocks, BLOCK_SIZE>>>(d_fluxs, d_dfluxs, num_elements);
    cudaDeviceSynchronize();
}

__device__ __forceinline__ int xi_eta_idx(const int xi, const int eta)
{
    const int i = xi + 1;
    const int j = eta + 1;
    const int id = j * 3 + i;
    constexpr int idx[] = {0, 1, 2, 7, 8, 3, 6, 5, 4};
    return idx[id];
}

__device__ __forceinline__ Vec4 normal_flux(const Flux &flux, const Point &normal)
{
    return flux.f * normal.x + flux.g * normal.y;
}

__global__ void rhs_kernel(const Cell *d_elements,
                           const Face *d_faces,
                           const Flux *d_fluxs,
                           const Flux *d_dfluxs,
                           const Vec4 *d_face_fluxs,
                           const JMatrix2d *d_jacobian_invT,
                           const double *d_jacobian_det,
                           const double *d_jacobian_face,
                           Vec4 *d_rhs,
                           const int num_elements)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements)
    {
        return;
    }

    const Cell &cell = d_elements[i];
    const Flux *flux = d_fluxs + i * 9;
    const Flux *dflux = d_dfluxs + i * 9;
    const Face &bottom_face = d_faces[cell.faceIds[0]];
    const Face &right_face = d_faces[cell.faceIds[1]];
    const Face &top_face = d_faces[cell.faceIds[2]];
    const Face &left_face = d_faces[cell.faceIds[3]];
    const Vec4 *bottom_face_flux = d_face_fluxs + cell.faceIds[0] * 3;
    const Vec4 *right_face_flux = d_face_fluxs + cell.faceIds[1] * 3;
    const Vec4 *top_face_flux = d_face_fluxs + cell.faceIds[2] * 3;
    const Vec4 *left_face_flux = d_face_fluxs + cell.faceIds[3] * 3;
    const double bottom_face_jacobian = d_jacobian_face[cell.faceIds[0]];
    const double right_face_jacobian = d_jacobian_face[cell.faceIds[1]];
    const double top_face_jacobian = d_jacobian_face[cell.faceIds[2]];
    const double left_face_jacobian = d_jacobian_face[cell.faceIds[3]];
    const JMatrix2d *jacobian_invT = d_jacobian_invT + i * 9;
    const double *jacobian_det = d_jacobian_det + i * 9;
    Vec4 *rhs = d_rhs + i * 9;

    for (int j = 0; j < 9; ++j)
    {
        const int xi = idx_xi(j);
        const int eta = idx_eta(j);
        constexpr double DG2R[3] = {1.5, -0.75, 4.5};
        Flux dflux_corr;
        dflux_corr.f = dflux[j].f + ((-left_face_flux[eta + 1] - normal_flux(flux[xi_eta_idx(-1, eta)], -left_face.normal)) *
                                         DG2R[-xi + 1] * left_face_jacobian +
                                     (right_face_flux[eta + 1] - normal_flux(flux[xi_eta_idx(1, eta)], right_face.normal)) *
                                         DG2R[xi + 1] * right_face_jacobian) /
                                        jacobian_det[j];
        dflux_corr.g = dflux[j].g + ((-bottom_face_flux[xi + 1] - normal_flux(flux[xi_eta_idx(xi, -1)], -bottom_face.normal)) *
                                         DG2R[-eta + 1] * bottom_face_jacobian +
                                     (top_face_flux[xi + 1] - normal_flux(flux[xi_eta_idx(xi, 1)], top_face.normal)) *
                                         DG2R[eta + 1] * top_face_jacobian) /
                                        jacobian_det[j];
        const Flux div_flux = jacobian_invT[j] * dflux_corr;
        rhs[j] = -(div_flux.f + div_flux.g);
    }
}

void calculate_rhs(const Cell *d_elements,
                   const Face *d_faces,
                   const Flux *d_fluxs,
                   const Flux *d_dfluxs,
                   const Vec4 *d_face_fluxs,
                   const JMatrix2d *d_jacobian_invT,
                   const double *d_jacobian_det,
                   const double *d_jacobian_face,
                   Vec4 *d_rhs,
                   const int num_elements)
{
    const int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rhs_kernel<<<num_blocks, BLOCK_SIZE>>>(d_elements, d_faces, d_fluxs, d_dfluxs, d_face_fluxs, d_jacobian_invT,
                                           d_jacobian_det, d_jacobian_face, d_rhs, num_elements);
    cudaDeviceSynchronize();
}

__global__ void time_forward_kernel(const Vec4 *d_src_nodes, Vec4 *d_dst_nodes, const Vec4 *d_rhs, const double dt,
                                    const int numElements)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElements)
    {
        return;
    }
    const Vec4 *src_nodes = d_src_nodes + i * 9;
    const Vec4 *rhs = d_rhs + i * 9;
    Vec4 *dst_nodes = d_dst_nodes + i * 9;
    for (int j = 0; j < 9; ++j)
    {
        dst_nodes[j] = src_nodes[j] + rhs[j] * dt;
    }
}

void calculate_time_forward(const Vec4 *d_src_nodes, Vec4 *d_dst_nodes, const Vec4 *d_rhs, const double dt,
                            const int numElements)
{
    const int num_blocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    time_forward_kernel<<<num_blocks, BLOCK_SIZE>>>(d_src_nodes, d_dst_nodes, d_rhs, dt, numElements);
    cudaDeviceSynchronize();
}
