#include <cmath>

#include "nodes_init.h"
#include "euler_eq.cuh"
#include "shape_f.h"

void init_fws_nodes(const Mesh &mesh, Vec4 *nodes, const Vec4 &init_P, double gamma)
{
    const Vec4 Q0 = toConservative(init_P, gamma);
    for (int i = 0; i < mesh.numElements; ++i)
    {
        if (!mesh.elements[i].isValid)
            continue;
        for (int j = 0; j < 9; ++j)
        {
            nodes[i * 9 + j] = Q0;
        }
    }
}

void init_inf_nodes(const Mesh &mesh, Vec4 *nodes, const double u_inf, const double v_inf, const double beta, const double gamma)
{
    const double cx = mesh.width / 2;
    const double cy = mesh.height / 2;
    auto getP = [&](int cellId, int nodeId) -> Vec4
    {
        auto [xi, eta] = gll_2d(nodeId);
        auto [x, y] = interpolate({mesh.vertices[mesh.elements[cellId].vertexIds[0]],
                                   mesh.vertices[mesh.elements[cellId].vertexIds[1]],
                                   mesh.vertices[mesh.elements[cellId].vertexIds[2]],
                                   mesh.vertices[mesh.elements[cellId].vertexIds[3]]},
                                  xi, eta);
        double r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
        double T = 1 - (gamma - 1) * beta * beta / (8 * gamma * M_PI * M_PI) * std::exp(1 - r * r);
        double rho = std::pow(T, 1.0 / (gamma - 1.0));
        double u = u_inf - beta / (2 * M_PI) * std::exp((1 - r * r) / 2) * (y - cy);
        double v = v_inf + beta / (2 * M_PI) * std::exp((1 - r * r) / 2) * (x - cx);
        double p = std::pow(rho, gamma);

        return {rho, u, v, p};
    };
    for (int i = 0; i < mesh.numElements; ++i)
    {
        if (!mesh.elements[i].isValid)
            continue;
        for (int j = 0; j < 9; ++j)
        {
            nodes[i * 9 + j] = toConservative(getP(i, j), gamma);
        }
    }
}
