#include <gtest/gtest.h>

#include "mesh_init.h"
#include "nodes_init.h"
#include "flux_utils.h"
#include "jacobian.cuh"
#include "euler_eq.cuh"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(FLUX, phy_flux)
{
    Mesh mesh(2, 2, 2, 2);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    Flux *h_fluxs = new Flux[mesh.numElements * 9];
    Vec4 *d_nodes = nullptr;
    Flux *d_fluxs = nullptr;
    cudaMalloc(&d_nodes, sizeof(Vec4) * mesh.numElements * 9);
    cudaMalloc(&d_fluxs, sizeof(Flux) * mesh.numElements * 9);
    init_fws_mesh(mesh, 0, 0);
    init_fws_nodes(mesh, h_nodes, Vec4(1.4, 3.0, 0.0, 1.0), GAMMA);
    cudaMemcpy(d_nodes, h_nodes, sizeof(Vec4) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    calculate_physical_flux(d_nodes, d_fluxs, mesh.numElements, GAMMA);
    cudaMemcpy(h_fluxs, d_fluxs, sizeof(Flux) * mesh.numElements * 9, cudaMemcpyDeviceToHost);
    EXPECT_DOUBLE_EQ(h_fluxs[0].f[0], 4.2);
    EXPECT_DOUBLE_EQ(h_fluxs[0].f[1], 13.6);
    EXPECT_DOUBLE_EQ(h_fluxs[0].f[2], 0.0);
    EXPECT_DOUBLE_EQ(h_fluxs[0].f[3], 29.4);
    cudaFree(d_nodes);
    cudaFree(d_fluxs);
    delete[] h_nodes;
    delete[] h_fluxs;
}

TEST(FLUX, num_flux)
{
    Mesh mesh(2, 2, 2, 2);
    const Vec4 init_P(1.4, 3.0, 0.0, 1.0);
    const Vec4 init_Q = toConservative(init_P, GAMMA);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    Vec4 *h_face_fluxs = new Vec4[mesh.numFaces * 3];
    init_fws_mesh(mesh, 0, 0);
    init_fws_nodes(mesh, h_nodes, init_P, GAMMA);
    // ---------------------------------------------------
    Face *d_faces = nullptr;
    Vec4 *d_nodes = nullptr;
    Flux *d_fluxs = nullptr;
    Vec4 *d_face_fluxs = nullptr;
    cudaMalloc(&d_faces, sizeof(Face) * mesh.numFaces);
    cudaMalloc(&d_nodes, sizeof(Vec4) * mesh.numElements * 9);
    cudaMalloc(&d_fluxs, sizeof(Flux) * mesh.numElements * 9);
    cudaMalloc(&d_face_fluxs, sizeof(Vec4) * mesh.numFaces * 3);
    // ---------------------------------------------------
    cudaMemcpy(d_faces, mesh.faces, sizeof(Face) * mesh.numFaces, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes, h_nodes, sizeof(Vec4) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    calculate_physical_flux(d_nodes, d_fluxs, mesh.numElements, GAMMA);
    calculate_face_num_flux(d_faces, d_nodes, d_fluxs, d_face_fluxs, init_Q, mesh.nx, mesh.ny, GAMMA);
    cudaMemcpy(h_face_fluxs, d_face_fluxs, sizeof(Vec4) * mesh.numFaces * 3, cudaMemcpyDeviceToHost);
    // ---------------------------------------------------
    EXPECT_DOUBLE_EQ(h_face_fluxs[0][0], 4.2);
    EXPECT_DOUBLE_EQ(h_face_fluxs[0][1], 13.6);
    EXPECT_DOUBLE_EQ(h_face_fluxs[0][2], 0.0);
    EXPECT_DOUBLE_EQ(h_face_fluxs[0][3], 29.4);
    // ---------------------------------------------------
    cudaFree(d_nodes);
    cudaFree(d_fluxs);
    cudaFree(d_face_fluxs);
    delete[] h_nodes;
    delete[] h_face_fluxs;
}

TEST(FLUX, div_flux)
{
    Mesh mesh(2, 2, 2, 2);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    Flux *h_dfluxs = new Flux[mesh.numElements * 9];
    init_fws_mesh(mesh, 1, 1);
    init_fws_nodes(mesh, h_nodes, Vec4(1.4, 3.0, 0.0, 1.0), GAMMA);
    // ---------------------------------------------------
    Vec4 *d_nodes = nullptr;
    Flux *d_fluxs = nullptr;
    Flux *d_dfluxs = nullptr;
    cudaMalloc(&d_nodes, sizeof(Vec4) * mesh.numElements * 9);
    cudaMalloc(&d_fluxs, sizeof(Flux) * mesh.numElements * 9);
    cudaMalloc(&d_dfluxs, sizeof(Flux) * mesh.numElements * 9);
    // ---------------------------------------------------
    cudaMemcpy(d_nodes, h_nodes, sizeof(Vec4) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    calculate_physical_flux(d_nodes, d_fluxs, mesh.numElements, GAMMA);
    calculate_physical_div_flux(d_fluxs, d_dfluxs, mesh.numElements);
    cudaMemcpy(h_dfluxs, d_dfluxs, sizeof(Flux) * mesh.numElements * 9, cudaMemcpyDeviceToHost);
    // ---------------------------------------------------
    EXPECT_NEAR(h_dfluxs[0].f[0], 0.0, 1e-6);
    EXPECT_NEAR(h_dfluxs[0].f[1], 0.0, 1e-6);
    EXPECT_NEAR(h_dfluxs[0].f[2], 0.0, 1e-6);
    EXPECT_NEAR(h_dfluxs[0].f[3], 0.0, 1e-6);
    // ---------------------------------------------------
    cudaFree(d_nodes);
    cudaFree(d_fluxs);
    cudaFree(d_dfluxs);
    delete[] h_nodes;
    delete[] h_dfluxs;
}

TEST(FLUX, rhs)
{
    Mesh mesh(1, 1, 1, 1);
    const Vec4 init_P(1.4, 3.0, 0.0, 1.0);
    const Vec4 init_Q = toConservative(init_P, GAMMA);
    Vec4 *h_nodes = new Vec4[mesh.numElements * 9];
    auto *h_jacobian_invT = new JMatrix2d[mesh.numElements * 9];
    auto *h_jacobian_det = new double[mesh.numElements * 9];
    auto *h_jacobian_face = new double[mesh.numFaces];
    init_fws_mesh(mesh, 0, 0);
    init_fws_nodes(mesh, h_nodes, init_P, GAMMA);
    calculate_jacobian_cell(mesh, h_jacobian_invT, h_jacobian_det);
    calculate_jacobian_face(mesh, h_jacobian_face);
    // ---------------------------------------------------
    Vec4 *d_nodes = nullptr;
    Flux *d_fluxs = nullptr;
    Flux *d_dfluxs = nullptr;
    Cell *d_elements = nullptr;
    Face *d_faces = nullptr;
    Vec4 *d_face_fluxs = nullptr;
    JMatrix2d *d_jacobian_invT = nullptr;
    double *d_jacobian_det = nullptr;
    double *d_jacobian_face = nullptr;
    Vec4 *d_rhs = nullptr;
    cudaMalloc(&d_nodes, sizeof(Vec4) * mesh.numElements * 9);
    cudaMalloc(&d_fluxs, sizeof(Flux) * mesh.numElements * 9);
    cudaMalloc(&d_dfluxs, sizeof(Flux) * mesh.numElements * 9);
    cudaMalloc(&d_elements, sizeof(Cell) * mesh.numElements);
    cudaMalloc(&d_faces, sizeof(Face) * mesh.numFaces);
    cudaMalloc(&d_face_fluxs, sizeof(Vec4) * mesh.numFaces * 3);
    cudaMalloc(&d_jacobian_invT, sizeof(JMatrix2d) * mesh.numElements * 9);
    cudaMalloc(&d_jacobian_det, sizeof(double) * mesh.numElements * 9);
    cudaMalloc(&d_jacobian_face, sizeof(double) * mesh.numFaces);
    cudaMalloc(&d_rhs, sizeof(Vec4) * mesh.numElements * 9);
    // ---------------------------------------------------
    cudaMemcpy(d_elements, mesh.elements, sizeof(Cell) * mesh.numElements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, mesh.faces, sizeof(Face) * mesh.numFaces, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nodes, h_nodes, sizeof(Vec4) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobian_invT, h_jacobian_invT, sizeof(JMatrix2d) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobian_det, h_jacobian_det, sizeof(double) * mesh.numElements * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_jacobian_face, h_jacobian_face, sizeof(double) * mesh.numFaces, cudaMemcpyHostToDevice);
    // ---------------------------------------------------
    calculate_physical_flux(d_nodes, d_fluxs, mesh.numElements, GAMMA);
    calculate_physical_div_flux(d_fluxs, d_dfluxs, mesh.numElements);
    calculate_face_num_flux(d_faces, d_nodes, d_fluxs, d_face_fluxs, init_Q, mesh.nx, mesh.ny, GAMMA);
    calculate_rhs(d_elements, d_faces, d_fluxs, d_dfluxs, d_face_fluxs, d_jacobian_invT, d_jacobian_det, d_jacobian_face, d_rhs, mesh.numElements);
    // ---------------------------------------------------
    cudaMemcpy(h_nodes, d_rhs, sizeof(Vec4) * mesh.numElements * 9, cudaMemcpyDeviceToHost);
    // ---------------------------------------------------
    EXPECT_DOUBLE_EQ(h_nodes[8][0], 0.0);
    EXPECT_DOUBLE_EQ(h_nodes[8][1], 0.0);
    EXPECT_DOUBLE_EQ(h_nodes[8][2], 0.0);
    EXPECT_DOUBLE_EQ(h_nodes[8][3], 0.0);
    // ---------------------------------------------------
    cudaFree(d_nodes);
    cudaFree(d_fluxs);
    cudaFree(d_dfluxs);
    cudaFree(d_elements);
    cudaFree(d_faces);
    cudaFree(d_face_fluxs);
    cudaFree(d_jacobian_invT);
    cudaFree(d_jacobian_det);
    cudaFree(d_jacobian_face);
    cudaFree(d_rhs);
    delete[] h_nodes;
    delete[] h_jacobian_invT;
    delete[] h_jacobian_det;
    delete[] h_jacobian_face;
}
