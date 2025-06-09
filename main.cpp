#include <iostream>

#include "solver.h"
#include "mesh_init.h"
#include "nodes_init.h"
#include "visualize.h"

int main()
{
    Mesh mesh(10, 10, 10.0, 10.0);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    init_inf_mesh(mesh);
    init_inf_nodes(mesh, h_nodes, 1, 1, 5, GAMMA);
    FREulerSolver solver(mesh, h_nodes);
    std::string fileName = "output/output_" + std::to_string(0) + ".vtu";
    writeVTU(fileName.data(), h_nodes, mesh, GAMMA);
    for (int i = 1; i <= 10000; ++i)
    {
        solver.advance(0.001);
        if (i % 100 == 0)
        {
            std::cout << "Current time: " << solver.getCurrentTime() << std::endl;
            solver.getNodes(h_nodes);
            fileName = "output/output_" + std::to_string(i / 100) + ".vtu";
            writeVTU(fileName.data(), h_nodes, mesh, GAMMA);
        }
    }
    delete[] h_nodes;
    return 0;
}