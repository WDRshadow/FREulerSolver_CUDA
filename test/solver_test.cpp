#include <gtest/gtest.h>

#include "mesh_init.h"
#include "nodes_init.h"
#include "jacobian.cuh"
#include "solver.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(SOLVER, runtime)
{
    Mesh mesh(2, 2, 2, 2);
    const Vec4 init_P(1.4, 3.0, 0.0, 1.0);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    init_fws_mesh(mesh, 1, 1);
    init_fws_nodes(mesh, h_nodes, init_P, GAMMA);
    FREulerSolver solver(mesh, h_nodes);
    solver.set_fws_bc(init_P);
    solver.advance(0.001);
    EXPECT_DOUBLE_EQ(solver.getCurrentTime(), 0.001);
}

TEST(SOLVER, limiter)
{
    Mesh mesh(2, 2, 2, 2);
    const Vec4 init_P(1.4, 3.0, 0.0, 1.0);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    init_fws_mesh(mesh, 1, 1);
    init_fws_nodes(mesh, h_nodes, init_P, GAMMA);
    FREulerSolver solver(mesh, h_nodes);
    solver.set_fws_bc(init_P);
    solver.set_tvb_limiter();
    solver.advance(0.001);
    EXPECT_DOUBLE_EQ(solver.getCurrentTime(), 0.001);
}
