#include <iostream>

#include "solver.h"
#include "mesh_init.h"
#include "nodes_init.h"
#include "visualize.h"

int main()
{
    Mesh mesh(30, 10, 3.0, 1.0);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    Vec4 init_P(1.4, 3.0, 0.0, 1.0);
    init_fws_mesh(mesh, 24, 2);
    init_fws_nodes(mesh, h_nodes, init_P, GAMMA);
    FREulerSolver solver(mesh, h_nodes);
    solver.set_tvb_limiter();
    std::string fileName = "output/output_" + std::to_string(0) + ".vtu";
    writeVTU(fileName.data(), h_nodes, mesh, GAMMA);
    for (int i = 1; i <= 100; ++i)
    {
        solver.advance(0.00001);
        if (i % 1 == 0)
        {
            std::cout << "Current time: " << solver.getCurrentTime() << std::endl;
            solver.getNodes(h_nodes);
            fileName = "output/output_" + std::to_string(i / 1) + ".vtu";
            writeVTU(fileName.data(), h_nodes, mesh, GAMMA);
        }
    }
    delete[] h_nodes;
    return 0;
}