#include <gtest/gtest.h>

#include "jacobian.cuh"
#include "mesh_init.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(JACOBIAN, face)
{
    Mesh mesh(2, 2, 1.0, 1.0);
    init_fws_mesh(mesh, 0, 0);
    double *mesh_jacobian_face = new double[mesh.numFaces];
    calculate_jacobian_face(mesh, mesh_jacobian_face);
    for (int i = 0; i < mesh.numFaces; ++i)
    {
        EXPECT_DOUBLE_EQ(mesh_jacobian_face[i], 0.25);
    }
    delete[] mesh_jacobian_face;
}

TEST(JACOBIAN, cell)
{
    Mesh mesh(1, 1, 1.0, 1.0);
    init_fws_mesh(mesh, 0, 0);
    Matrix2d *mesh_jacobian_cell_invT = new Matrix2d[mesh.numElements * 9];
    double *mesh_jacobian_cell_det = new double[mesh.numElements * 9];
    calculate_jacobian_cell(mesh, mesh_jacobian_cell_invT, mesh_jacobian_cell_det);
    for (int i = 0; i < mesh.numElements * 9; ++i)
    {
        EXPECT_DOUBLE_EQ(mesh_jacobian_cell_invT[i].J[0][0], 2);
        EXPECT_DOUBLE_EQ(mesh_jacobian_cell_invT[i].J[0][1], 0);
        EXPECT_DOUBLE_EQ(mesh_jacobian_cell_invT[i].J[1][0], 0);
        EXPECT_DOUBLE_EQ(mesh_jacobian_cell_invT[i].J[1][1], 2);
        EXPECT_DOUBLE_EQ(mesh_jacobian_cell_det[i], 0.25);
    }
    delete[] mesh_jacobian_cell_invT;
    delete[] mesh_jacobian_cell_det;
}
