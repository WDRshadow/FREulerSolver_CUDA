#include <gtest/gtest.h>

#include "nodes_init.h"
#include "mesh_init.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(NODES, fws)
{
    Mesh mesh(2, 2, 1.0, 1.0);
    init_fws_mesh(mesh, 1, 1);
    CellNode* cell_nodes = new CellNode[mesh.numElements];
    init_fws_nodes(mesh, cell_nodes, Vec4(1.0, 0.0, 0.0, 1.0), GAMMA);
    EXPECT_DOUBLE_EQ(cell_nodes[0].nodes[0][0], 1.0);
    EXPECT_DOUBLE_EQ(cell_nodes[1].nodes[2][1], 0.0);
    EXPECT_DOUBLE_EQ(cell_nodes[2].nodes[4][2], 0.0);
    EXPECT_DOUBLE_EQ(cell_nodes[3].nodes[6][3], 2.5);
}

TEST(NODES, inf)
{
    Mesh mesh(10, 10, 10.0, 10.0);
    init_inf_mesh(mesh);
    CellNode* cell_nodes = new CellNode[mesh.numElements];
    init_inf_nodes(mesh, cell_nodes, 0, 0, 5.0, GAMMA);
    EXPECT_NEAR(cell_nodes[0].nodes[0][0], 1.0, 1e-6);
    EXPECT_NEAR(cell_nodes[0].nodes[0][1], 0.0, 1e-6);
    EXPECT_NEAR(cell_nodes[0].nodes[0][2], 0.0, 1e-6);
    EXPECT_NEAR(cell_nodes[0].nodes[0][3], 2.5, 1e-6);
}
