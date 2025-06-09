#include <gtest/gtest.h>

#include "mesh_init.h"
#include "nodes_init.h"
#include "visualize.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(VISUALIZE, test)
{
    Mesh mesh(2, 2, 2, 2);
    const Vec4 init_P(1.4, 3.0, 0.0, 1.0);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    init_fws_mesh(mesh, 1, 1);
    init_fws_nodes(mesh, h_nodes, init_P, GAMMA);
    writeVTU("test_output.vtu", h_nodes, mesh, GAMMA);
}
