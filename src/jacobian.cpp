#include "jacobian.cuh"

JMatrix2d jacobian_cell(const Mesh &mesh, const int cellId, const int nodeId)
{
    const auto &cell = mesh.elements[cellId];
    const auto [xi, eta] = gll_2d(nodeId);
    JMatrix2d J{};
    for (int i = 0; i < 4; ++i)
    {
        auto [dN_dxi, dN_deta] = dshape(i, xi, eta);
        J.J[0][0] += mesh.vertices[cell.vertexIds[i]].x * dN_dxi;
        J.J[0][1] += mesh.vertices[cell.vertexIds[i]].x * dN_deta;
        J.J[1][0] += mesh.vertices[cell.vertexIds[i]].y * dN_dxi;
        J.J[1][1] += mesh.vertices[cell.vertexIds[i]].y * dN_deta;
    }
    return J;
}

double jacobian_face(const Mesh &mesh, const int faceId)
{
    constexpr int left_idx[] = {0, 3};
    constexpr int right_idx[] = {1, 2};
    constexpr int bottom_idx[] = {0, 1};
    constexpr int top_idx[] = {3, 2};
    Point p1, p2;
    if (faceId < (mesh.nx + 1) * mesh.ny)
    {
        const int i = faceId % (mesh.nx + 1);
        const int j = faceId / (mesh.nx + 1);
        if (i != mesh.nx)
        {
            const Cell &cell = mesh.elements[j * mesh.nx + i];
            p1 = mesh.vertices[cell.vertexIds[left_idx[0]]];
            p2 = mesh.vertices[cell.vertexIds[left_idx[1]]];
        }
        else
        {
            const Cell &cell = mesh.elements[j * mesh.nx + i - 1];
            p1 = mesh.vertices[cell.vertexIds[right_idx[0]]];
            p2 = mesh.vertices[cell.vertexIds[right_idx[1]]];
        }
    }
    else
    {
        const int i = (faceId - (mesh.nx + 1) * mesh.ny) % mesh.nx;
        const int j = (faceId - (mesh.nx + 1) * mesh.ny) / mesh.nx;
        if (j != mesh.ny)
        {
            const Cell &cell = mesh.elements[j * mesh.nx + i];
            p1 = mesh.vertices[cell.vertexIds[bottom_idx[0]]];
            p2 = mesh.vertices[cell.vertexIds[bottom_idx[1]]];
        }
        else
        {
            const Cell &cell = mesh.elements[(j - 1) * mesh.nx + i];
            p1 = mesh.vertices[cell.vertexIds[top_idx[0]]];
            p2 = mesh.vertices[cell.vertexIds[top_idx[1]]];
        }
    }
    double L = std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    return 0.5 * L;
}

void calculate_jacobian_face(const Mesh &mesh, double *mesh_jacobian_face)
{
    for (int i = 0; i < mesh.numFaces; ++i)
    {
        mesh_jacobian_face[i] = jacobian_face(mesh, i);
    }
}

void calculate_jacobian_cell(const Mesh &mesh, JMatrix2d *mesh_jacobian_cell_invT, double *mesh_jacobian_cell_det)
{
    for (int i = 0; i < mesh.numElements; ++i)
    {
        for (int j = 0; j < 9; ++j)
        {
            const auto jacobian = jacobian_cell(mesh, i, j);
            mesh_jacobian_cell_det[i * 9 + j] = jacobian.det();
            mesh_jacobian_cell_invT[i * 9 + j] = jacobian.inv().T();
        }
    }
}