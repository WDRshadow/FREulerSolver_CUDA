#include <gtest/gtest.h>

#include "limiter.h"
#include "euler_eq.cuh"
#include "mesh_init.h"
#include "nodes_init.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(LIMITER, runtime)
{
    Mesh mesh(1, 1, 1, 1);
    Vec4 init_P{1.4, 3.0, 0.0, 1.0};
    Vec4 init_U = toConservative(init_P, GAMMA);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    init_fws_mesh(mesh, 0, 0);
    init_fws_nodes(mesh, h_nodes, init_P, GAMMA);
    // ---------------------------------------------------
    Cell *d_cells = nullptr;
    Face *d_faces = nullptr;
    Vec4 *d_nodes = nullptr;
    Vec4 *d_new_nodes = nullptr;
    cudaMalloc(&d_cells, sizeof(Cell) * mesh.numElements);
    cudaMalloc(&d_faces, sizeof(Face) * mesh.numFaces);
    cudaMalloc(&d_nodes, sizeof(Vec4) * mesh.numElements * 9);
    cudaMalloc(&d_new_nodes, sizeof(Vec4) * mesh.numElements * 9);
    // ---------------------------------------------------
    cudaMemcpy(d_cells, mesh.elements, sizeof(Cell) * mesh.numElements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, mesh.faces, sizeof(Face) * mesh.numFaces, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes, h_nodes, sizeof(Vec4) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    // ---------------------------------------------------
    tvd_limiter(d_cells, d_faces, d_nodes, d_new_nodes, mesh.numElements, init_U, GAMMA, 2, 2, 0.0);
    // ---------------------------------------------------
    cudaMemcpy(h_nodes, d_new_nodes, sizeof(Vec4) * mesh.numElements * 9, cudaMemcpyDeviceToHost);
    // ---------------------------------------------------
    EXPECT_NEAR(h_nodes[0][0], init_U[0], 1e-10);
    EXPECT_NEAR(h_nodes[0][1], init_U[1], 1e-10);
    EXPECT_NEAR(h_nodes[0][2], init_U[2], 1e-10);
    EXPECT_NEAR(h_nodes[0][3], init_U[3], 1e-10);
    // ---------------------------------------------------
    cudaFree(d_cells);
    cudaFree(d_faces);
    cudaFree(d_nodes);
    cudaFree(d_new_nodes);
    delete[] h_nodes;
}