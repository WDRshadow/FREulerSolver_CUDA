#include <iostream>

#include "solver.h"
#include "mesh_init.h"
#include "nodes_init.h"
#include "visualize.h"
#include "parser.h"

int main()
{
    CaseConfig caseCfg;
    MeshConfig meshCfg;
    ICConfig icCfg;
    Vec4 bcCfg;
    SolverConfig solverCfg;
    LimiterConfig limiterCfg;
    VisualizeConfig visCfg;

    ensureConfigFile("config.ini");
    parseConfig("config.ini", caseCfg, meshCfg, icCfg, bcCfg, solverCfg, limiterCfg, visCfg);

    if (limiterCfg.enable)
    {
        ensureOutputFolder("output");
    }

    int method;
    if (solverCfg.method == "Euler")
        method = EULER;
    else if (solverCfg.method == "RK4")
        method = RK4;

    Mesh mesh(meshCfg.nx, meshCfg.ny, meshCfg.width, meshCfg.height);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    if (caseCfg.type == "fws")
    {
        init_fws_mesh(mesh, meshCfg.stepNX, meshCfg.stepNY);
        init_fws_nodes(mesh, h_nodes, icCfg.fws, GAMMA);
    }
    else if (caseCfg.type == "inf")
    {
        init_inf_mesh(mesh);
        init_inf_nodes(mesh, h_nodes, icCfg.inf_u, icCfg.inf_v, icCfg.inf_beta, GAMMA);
    }
    FREulerSolver solver(mesh, h_nodes);
    solver.set_fws_bc(bcCfg);
    if (limiterCfg.enable)
    {
        solver.set_tvb_limiter(limiterCfg.M);
    }
    for (int i = 0; i <= solverCfg.steps; ++i)
    {
        if (visCfg.enable && i % visCfg.gap == 0)
        {
            std::cout << "Current time: " << solver.getCurrentTime() << std::endl;
            solver.getNodes(h_nodes);
            std::string fileName = "output/output_" + std::to_string(i / visCfg.gap) + ".vtu";
            writeVTU(fileName.data(), h_nodes, mesh, GAMMA);
        }
        if (i == solverCfg.steps)
            break;
        solver.advance(solverCfg.step_size, method);
    }
    delete[] h_nodes;
    return 0;
}