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
    Vec4 *nodes = new Vec4[mesh.numElements * 9];
    init_fws_nodes(mesh, nodes, Vec4(1.0, 0.0, 0.0, 1.0), GAMMA);
    EXPECT_DOUBLE_EQ(nodes[0][0], 1.0);
    EXPECT_DOUBLE_EQ(nodes[10][1], 0.0);
    EXPECT_DOUBLE_EQ(nodes[20][2], 0.0);
    EXPECT_DOUBLE_EQ(nodes[30][3], 2.5);
}

TEST(NODES, inf)
{
    Mesh mesh(10, 10, 10.0, 10.0);
    init_inf_mesh(mesh);
    Vec4 *nodes = new Vec4[mesh.numElements * 9];
    init_inf_nodes(mesh, nodes, 0, 0, 5.0, GAMMA);
    EXPECT_NEAR(nodes[0][0], 1.0, 1e-6);
    EXPECT_NEAR(nodes[0][1], 0.0, 1e-6);
    EXPECT_NEAR(nodes[0][2], 0.0, 1e-6);
    EXPECT_NEAR(nodes[0][3], 2.5, 1e-6);
}
