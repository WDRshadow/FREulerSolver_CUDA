#include <gtest/gtest.h>

#include "mesh_init.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(MESH, fws)
{
    Mesh mesh(2, 2, 1.0, 1.0);
    init_fws_mesh(mesh, 1, 1);
    EXPECT_DOUBLE_EQ(mesh.vertices[4].y, 0.5);
    EXPECT_EQ(mesh.elements[3].faceIds[2], 11);
    EXPECT_EQ(mesh.faces[9].leftCell, Y_WALL);
    EXPECT_EQ(mesh.faces[1].rightCell, X_WALL);
}

TEST(MESH, inf)
{
    Mesh mesh(2, 2, 1.0, 1.0);
    init_inf_mesh(mesh);
    EXPECT_DOUBLE_EQ(mesh.vertices[4].y, 0.5);
    EXPECT_EQ(mesh.elements[3].faceIds[2], 11);
    EXPECT_EQ(mesh.faces[0].leftCell, 1);
    EXPECT_EQ(mesh.faces[2].rightCell, 0);
}
