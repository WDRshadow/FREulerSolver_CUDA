#include <cmath>
#include <vector>

#include "mesh_init.h"
#include "flux_utils.h"

int idx(const int i, const int j, const int nx)
{
    return j * (nx + 1) + i;
}

void init_fws_mesh(Mesh &mesh, const int stepNX, const int stepNY)
{
    const int nx = mesh.nx;
    const int ny = mesh.ny;
    const double width = mesh.width;
    const double height = mesh.height;

    // vertices
    for (int j = 0; j <= ny; ++j)
    {
        for (int i = 0; i <= nx; ++i)
        {
            double x = width * i / nx;
            double y = height * j / ny;
            mesh.vertices[idx(i, j, nx)] = {x, y};
        }
    }

    // cell
    std::vector cellMap(nx * ny, -1);
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int cellId = j * nx + i;
            Cell &cell = mesh.elements[cellId];
            if (i >= nx - stepNX && j < stepNY)
            {
                cell.isValid = false;
            }
            else
            {
                cell.isValid = true;
                cellMap[cellId] = cellId;
            }
            cell.vertexIds[0] = idx(i, j, nx);
            cell.vertexIds[1] = idx(i + 1, j, nx);
            cell.vertexIds[2] = idx(i + 1, j + 1, nx);
            cell.vertexIds[3] = idx(i, j + 1, nx);
        }
    }

    // verticle faces
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i <= nx; ++i)
        {
            int faceId = j * (nx + 1) + i;
            Face &face = mesh.faces[faceId];
            int left = i == 0 ? INLET : cellMap[j * nx + (i - 1)];
            int right = i == nx ? OUTLET : cellMap[j * nx + i];
            if (left >= 0 && right < 0 && j < stepNY)
            {
                right = X_WALL;
            }
            const auto &v0 = mesh.vertices[idx(i, j, nx)];
            const auto &v1 = mesh.vertices[idx(i, j + 1, nx)];
            double dx_ = v1.x - v0.x;
            double dy_ = v1.y - v0.y;
            double norm = std::sqrt(dx_ * dx_ + dy_ * dy_);
            face.normal.x = dy_ / norm;
            face.normal.y = -dx_ / norm;
            face.leftCell = left;
            face.rightCell = right;
            if (left != -1)
                mesh.elements[left].faceIds[1] = faceId;
            if (right != -1)
                mesh.elements[right].faceIds[3] = faceId;
        }
    }

    // horizontal faces
    for (int j = 0; j <= ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int faceId = (nx + 1) * ny + j * nx + i;
            Face &face = mesh.faces[faceId];
            int down = j == 0 ? Y_WALL : cellMap[(j - 1) * nx + i];
            int up = j == ny ? Y_WALL : cellMap[j * nx + i];
            if (j == stepNY && i >= nx - stepNX)
            {
                down = Y_WALL;
            }
            const auto &v0 = mesh.vertices[idx(i, j, nx)];
            const auto &v1 = mesh.vertices[idx(i + 1, j, nx)];
            double dx_ = v1.x - v0.x;
            double dy_ = v1.y - v0.y;
            double norm = std::sqrt(dx_ * dx_ + dy_ * dy_);
            face.normal.x = -dy_ / norm;
            face.normal.y = dx_ / norm;
            face.leftCell = down;
            face.rightCell = up;
            if (down != -1)
                mesh.elements[down].faceIds[2] = faceId;
            if (up != -1)
                mesh.elements[up].faceIds[0] = faceId;
        }
    }
}

void init_inf_mesh(Mesh &mesh)
{
    const int nx = mesh.nx;
    const int ny = mesh.ny;
    const double width = mesh.width;
    const double height = mesh.height;

    // vertices
    for (int j = 0; j <= ny; ++j)
    {
        for (int i = 0; i <= nx; ++i)
        {
            double x = width * i / nx;
            double y = height * j / ny;
            mesh.vertices[idx(i, j, nx)] = {x, y};
        }
    }

    // cell
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int cellId = j * nx + i;
            Cell &cell = mesh.elements[cellId];
            cell.isValid = true;
            cell.vertexIds[0] = idx(i, j, nx);
            cell.vertexIds[1] = idx(i + 1, j, nx);
            cell.vertexIds[2] = idx(i + 1, j + 1, nx);
            cell.vertexIds[3] = idx(i, j + 1, nx);
        }
    }

    // vertical faces
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i <= nx; ++i)
        {
            int id = j * (nx + 1) + i;
            Face &face = mesh.faces[id];
            const auto &v0 = mesh.vertices[idx(i, j, nx)];
            const auto &v1 = mesh.vertices[idx(i, j + 1, nx)];
            double dx = v1.x - v0.x;
            double dy = v1.y - v0.y;
            double norm = std::sqrt(dx * dx + dy * dy);
            face.normal.x = dy / norm;
            face.normal.y = -dx / norm;
            if (i == 0 || i == nx)
            {
                face.leftCell = j * nx + (nx - 1);
                face.rightCell = j * nx + 0;
            }
            else
            {
                face.leftCell = j * nx + (i - 1);
                face.rightCell = j * nx + i;
            }
            if (face.leftCell != -1)
            {
                mesh.elements[face.leftCell].faceIds[1] = id;
            }
            if (face.rightCell != -1)
            {
                mesh.elements[face.rightCell].faceIds[3] = id;
            }
        }
    }

    // horizontal face
    for (int j = 0; j <= ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            int id = (nx + 1) * ny + nx * j + i;
            Face &face = mesh.faces[id];
            const auto &v0 = mesh.vertices[idx(i, j, nx)];
            const auto &v1 = mesh.vertices[idx(i + 1, j, nx)];
            double dx = v1.x - v0.x;
            double dy = v1.y - v0.y;
            double norm = std::sqrt(dx * dx + dy * dy);
            face.normal.x = -dy / norm;
            face.normal.y = dx / norm;
            if (j == 0 || j == ny)
            {
                face.leftCell = (ny - 1) * nx + i;
                face.rightCell = i;
            }
            else
            {
                face.leftCell = (j - 1) * nx + i;
                face.rightCell = j * nx + i;
            }
            if (face.leftCell != -1)
            {
                mesh.elements[face.leftCell].faceIds[2] = id;
            }
            if (face.rightCell != -1)
            {
                mesh.elements[face.rightCell].faceIds[0] = id;
            }
        }
    }
}