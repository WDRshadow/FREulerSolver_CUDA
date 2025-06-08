#include <gtest/gtest.h>

#include "mesh_init.h"
#include "nodes_init.h"
#include "flux_utils.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(NUM_FLUX, kernel)
{
    Mesh mesh(2, 2, 2, 2);
    init_fws_mesh(mesh, 1, 1);
    CellNode *init_cell_node = new CellNode[mesh.numElements];
    init_fws_nodes(mesh, init_cell_node, Vec4(1.4, 3.0, 0.0, 1.0), GAMMA);
    NumFluxCalculator num_flux_calculator(mesh, Vec4(1.4, 3.0, 0.0, 1.0), GAMMA);
    CellNode *d_cell_node = nullptr;
    cudaMalloc(&d_cell_node, sizeof(CellNode) * mesh.numElements);
    cudaMemcpy(d_cell_node, init_cell_node, sizeof(CellNode) * mesh.numElements, cudaMemcpyHostToDevice);
    FaceNumFlux *d_face_flux = nullptr;
    cudaMalloc(&d_face_flux, sizeof(FaceNumFlux) * mesh.numFaces);
    num_flux_calculator.calculate_face_num_flux(d_cell_node, d_face_flux);
    FaceNumFlux *h_face_flux = new FaceNumFlux[mesh.numFaces];
    cudaMemcpy(h_face_flux, d_face_flux, sizeof(FaceNumFlux) * mesh.numFaces, cudaMemcpyDeviceToHost);
    cudaFree(d_cell_node);
    cudaFree(d_face_flux);
    EXPECT_DOUBLE_EQ(h_face_flux[0].f[0][0], 4.2);
    EXPECT_DOUBLE_EQ(h_face_flux[0].f[0][1], 13.6);
    EXPECT_DOUBLE_EQ(h_face_flux[0].f[0][2], 0.0);
    EXPECT_DOUBLE_EQ(h_face_flux[0].f[0][3], 29.4);
}