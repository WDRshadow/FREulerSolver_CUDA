#include "flux_utils.h"
#include "euler_eq.cuh"

__device__ __forceinline__ double maxEig(const Vec4 &Q, const Point &normal, const double gamma)
{
    const Vec4 P = toPrimitive(Q, gamma);
    const double nx = normal.x;
    const double ny = normal.y;
    const double un = P[1] * nx + P[2] * ny;
    const double c = sqrt(gamma * P[3] / P[0]);
    return fabs(un) + c;
}

__device__ __forceinline__ Vec4 l_f_flux(const Vec4 &QL, const Vec4 &QR, const Point &normal, const double gamma)
{
    const double nx = normal.x, ny = normal.y;
    const double sL = maxEig(QL, normal, gamma);
    const double sR = maxEig(QR, normal, gamma);
    const double s = max(sL, sR);
    const Flux phy_flux_L = physicalFlux(QL, gamma);
    const Flux phy_flux_R = physicalFlux(QR, gamma);
    return (phy_flux_L.f * nx + phy_flux_L.g * ny + phy_flux_R.f * nx + phy_flux_R.g * ny) * 0.5 - (QR - QL) * 0.5 * s;
}

__device__ __forceinline__ void set_bc(const Vec4 &bc, const Vec4 *Q, const int bc_type, Vec4 *nodes)
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
    }
}

__device__ __forceinline__ void set_nodes(const CellNode &cell_node, Vec4 *nodes, const int idx[3])
{
    for (int j = 0; j < 3; ++j)
    {
        nodes[j] = cell_node.nodes[idx[j]];
    }
}

__global__ void face_num_flux_kernel(const Face *d_faces, const CellNode *d_cell_nodes, FaceNumFlux *d_face_flux, const int nx, const int ny, const Vec4 bc, const double gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int numVerticalFace = (nx + 1) * ny;
    const int numHorizontalFace = nx * (ny + 1);
    if (i >= numVerticalFace + numHorizontalFace)
    {
        return;
    }
    const Face &face = d_faces[i];
    const Point &normal = face.normal;
    constexpr int left_idx[] = {2, 3, 4};
    constexpr int right_idx[] = {0, 7, 6};
    constexpr int bottom_idx[] = {0, 1, 2};
    constexpr int top_idx[] = {6, 5, 4};
    if (face.leftCell < 0 && face.rightCell < 0)
    {
        return;
    }
    Vec4 l_nodes[3]{}, r_nodes[3]{};
    if (face.leftCell >= 0)
    {
        const CellNode &left_cell_node = d_cell_nodes[face.leftCell];
        if (i < numVerticalFace)
        {
            set_nodes(left_cell_node, l_nodes, left_idx);
        }
        else
        {
            set_nodes(left_cell_node, l_nodes, bottom_idx);
        }
    }
    if (face.rightCell >= 0)
    {
        const CellNode &right_cell_node = d_cell_nodes[face.rightCell];
        if (i < numVerticalFace)
        {
            set_nodes(right_cell_node, r_nodes, right_idx);
        }
        else
        {
            set_nodes(right_cell_node, r_nodes, top_idx);
        }
    }
    if (face.leftCell < 0)
    {
        set_bc(bc, r_nodes, face.leftCell, l_nodes);
    }
    if (face.rightCell < 0)
    {
        set_bc(bc, l_nodes, face.rightCell, r_nodes);
    }
    for (int j = 0; j < 3; ++j)
    {
        d_face_flux[i].f[j] = l_f_flux(l_nodes[j], r_nodes[j], normal, gamma);
    }
}

NumFluxCalculator::NumFluxCalculator(const Mesh &mesh, const Vec4 &bc_P, double gamma)
    : nx(mesh.nx), ny(mesh.ny), width(mesh.width), height(mesh.height),
      numVertices(mesh.numVertices), numFaces(mesh.numFaces), numElements(mesh.numElements),
      bc(toConservative(bc_P, gamma)), gamma(gamma)
{
    cudaMalloc(&d_vertices, numVertices * sizeof(Point));
    cudaMalloc(&d_faces, numFaces * sizeof(Face));
    cudaMalloc(&d_elements, numElements * sizeof(Cell));
    cudaMemcpy(d_vertices, mesh.vertices, numVertices * sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, mesh.faces, numFaces * sizeof(Face), cudaMemcpyHostToDevice);
    cudaMemcpy(d_elements, mesh.elements, numElements * sizeof(Cell), cudaMemcpyHostToDevice);
}

void NumFluxCalculator::calculate_face_num_flux(const CellNode *d_cell_nodes, FaceNumFlux *d_face_flux)
{
    constexpr int block_size = 512;
    const int num_blocks = (numFaces + block_size - 1) / block_size;
    face_num_flux_kernel<<<num_blocks, block_size>>>(d_faces, d_cell_nodes, d_face_flux, nx, ny, bc, gamma);
    cudaDeviceSynchronize();
}
